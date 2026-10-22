//  Records how the chat parser (common_chat_parse) handles native tool calls that are cut by max_tokens
//  or are malformed, for the tool-call formats the production Qwen models emit (UFH14-LB4, intake-2388).
//
//  Only the control case (d, a complete valid tool call parsed as final) is asserted. Every other case is a
//  RECORDED observation printed as one machine-readable RESULT line; it never fails the test.
//
//  No model, no network. The parser is generated from the chat templates in models/templates, so the
//  working directory must be the repository root (or pass the root as argv[1]):
//
//    ./build/bin/test-chat-parse-truncation            # from the repo root
//    ./build/bin/test-chat-parse-truncation /path/to/llama.cpp
//
//  ---------------------------------------------------------------------------------------------------
//  Server code path this test mirrors (traced in this tree; file:line):
//
//  * tools/server/server-task.cpp:164-175  task_result_state::update_chat_msg() calls
//      common_chat_parse(generated_text, is_partial, chat_parser_params) on the WHOLE generated text.
//  * tools/server/server-task.h:397-399    server_task_result_cmpl_final::update() always passes
//      is_partial = false. It does not look at the stop type, so a generation cut by max_tokens
//      (STOP_TYPE_LIMIT, finish_reason "length", server-task.cpp:445-454) is parsed exactly like an EOS stop.
//  * tools/server/server-task.cpp:1018-1023 server_task_result_cmpl_partial::update() passes is_partial = true
//      (streamed chunks only; the non-stream path never produces partial results).
//  * tools/server/server-queue.cpp:403-408 update() runs inside server_response_reader::next(), i.e. on the
//      HTTP handler thread, so an exception thrown by the parser escapes into the HTTP handler.
//  * common/chat.cpp:2859-2862 common_chat_parse() -> common_chat_peg_parse(); chat.cpp:2883 the PEG context is
//      ALWAYS lenient, so input that ends early yields NEED_MORE_INPUT (not FAIL) regardless of is_partial
//      (common/peg-parser.cpp:581, 711, 841). Only a real FAIL reaches chat.cpp:2891-2914: with is_partial and
//      result.end > 0 the partial AST is mapped and returned; otherwise it throws
//      std::runtime_error("The model produced output that does not match the expected <fmt> format").
//  * common/chat-peg-parser.cpp:337-343 JSON-form arguments are copied verbatim from the TOOL_ARGS node text;
//      chat-peg-parser.cpp:394-410 unterminated quotes/braces are closed ONLY when a TOOL_CLOSE node is seen,
//      so a call cut before its closing tag keeps its raw, unterminated arguments text.
//  * Non-stream: server-context.cpp:4935 rd.wait_for_all() -> exception leaves handle_completions_impl ->
//      tools/server/server.cpp:64-66 ex_wrapper catches std::exception as ERROR_TYPE_SERVER (std::invalid_argument
//      would be ERROR_TYPE_INVALID_REQUEST, server.cpp:59-62) -> server-common.cpp:33-35 code 500 ->
//      server.cpp:74-77 res->status = that code.
//  * Stream: the first result is read in the handler (server-context.cpp:4967, a throw there -> ex_wrapper -> 500),
//      then res->status = 200 (server-context.cpp:4997) and later results are read inside the set_next lambda
//      (server-context.cpp:5050). A throw there is caught at server-context.cpp:5097-5102 and sent as an SSE
//      `data: {"error": {"code": 500, ...}}` event, the stream ends without `data: [DONE]`; HTTP status stays 200.
//  * server-task.cpp:176 an EMPTY parse result does not replace the state message, and
//      server-task.cpp:445-452 an empty message falls back to returning the raw generated text as content.
//  ---------------------------------------------------------------------------------------------------
#include "chat.h"
#include "common.h"
#include "log.h"

#include <cstdio>
#include <exception>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

static std::string read_file(const std::string & path) {
    std::ifstream fs(path, std::ios_base::binary);
    if (!fs.is_open()) {
        throw std::runtime_error("cannot open template: " + path);
    }
    std::stringstream ss;
    ss << fs.rdbuf();
    return ss.str();
}

// escape for a double-quoted field on one line; truncated to max_len bytes
static std::string escape_field(const std::string & s, size_t max_len) {
    std::string out;
    for (size_t i = 0; i < s.size() && i < max_len; ++i) {
        const char c = s[i];
        if (c == '\\') {
            out += "\\\\";
        } else if (c == '"') {
            out += "\\\"";
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else if (static_cast<unsigned char>(c) < 0x20) {
            out += '?';
        } else {
            out += c;
        }
    }
    return out;
}

static const common_chat_tool weather_tool{
    /* .name = */ "get_weather",
    /* .description = */ "Get the weather forecast for a city",
    /* .parameters = */ R"({
        "type": "object",
        "properties": {
            "city": { "type": "string",  "description": "City name" },
            "days": { "type": "integer", "description": "Number of forecast days" }
        },
        "required": ["city"]
    })",
};

struct trunc_case {
    char        id;
    std::string desc;
    std::string text;
};

struct trunc_format {
    std::string             label;
    std::string             template_path;
    std::vector<trunc_case> cases;
};

// Hermes-style JSON tool calls: <tool_call>{"name": .., "arguments": {..}}</tool_call>
static std::vector<trunc_case> json_cases() {
    return {
        // (a) cut by max_tokens in the middle of a string argument value
        { 'a', "json_cut_mid_arguments",
          "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Par" },
        // (b) non-partial parse failure: malformed JSON inside a CLOSED tool call
        { 'b', "json_malformed_closed",
          "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\",,}}\n</tool_call>" },
        // (c) cut right after the opening tag
        { 'c', "json_cut_after_open_tag", "<tool_call>\n" },
        // (d) CONTROL: complete, valid tool call
        { 'd', "json_complete_valid",
          "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}\n</tool_call>" },
        // (e) content, then a closed call whose "arguments" is a JSON STRING holding an object (double-encoded
        //     arguments, a known model habit). Interesting because the mapper only adopts TOOL_ARGS text that
        //     starts with '{' (chat-peg-parser.cpp:341) and pads braces on close (chat-peg-parser.cpp:407), so
        //     this either fails the schema-typed grammar (throw) or silently becomes "{}" (argument loss).
        { 'e', "json_double_encoded_arguments",
          "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \"{\\\"city\\\": \\\"Paris\\\"}\"}\n</tool_call>" },
        // (f) cut after a complete arguments object but before the closing brace and tag
        { 'f', "json_cut_after_arguments_object",
          "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}" },
        // (g) closed, well-formed call to a tool that was not offered
        { 'g', "json_unknown_tool_closed",
          "<tool_call>\n{\"name\": \"x\", \"arguments\": {\"a\": 1}}\n</tool_call>" },
    };
}

// Qwen3-Coder / Qwen3.5+ XML tool calls: <tool_call><function=NAME><parameter=P>V</parameter></function></tool_call>
static std::vector<trunc_case> xml_cases() {
    return {
        // (a) cut by max_tokens in the middle of a parameter value
        { 'a', "xml_cut_mid_parameter",
          "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar" },
        // (b) non-partial parse failure: garbage between </function> and </tool_call> in a CLOSED call
        { 'b', "xml_malformed_closed",
          "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n}}garbage\n</tool_call>" },
        // (c) cut right after the opening tag
        { 'c', "xml_cut_after_open_tag", "<tool_call>\n" },
        // (d) CONTROL: complete, valid tool call
        { 'd', "xml_complete_valid",
          "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n"
          "</function>\n</tool_call>" },
        // (e) content, then a closed call whose integer-typed parameter carries a non-numeric value. Interesting
        //     because non-string values are spliced into the arguments text after only a quote normalisation
        //     (chat-peg-parser.cpp:381-382), so a schema-violating value can yield invalid JSON in a COMPLETE call
        //     (or the typed grammar rejects it and the final parse throws).
        { 'e', "xml_integer_param_non_numeric",
          "Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
          "<parameter=days>\nthree\n</parameter>\n</function>\n</tool_call>" },
        // (f) cut between two parameters (first one closed)
        { 'f', "xml_cut_between_parameters",
          "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n" },
        // (g) closed, well-formed call to a function that was not offered
        { 'g', "xml_unknown_tool_closed",
          "<tool_call>\n<function=x>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>" },
    };
}

// Static mapping from a parse outcome to what the client receives (see the trace at the top of the file).
static void derive_http(bool is_partial, bool threw, bool threw_invalid_argument,
                        std::string & http_nonstream, std::string & http_stream) {
    const std::string code = threw_invalid_argument ? "400" : "500";
    if (!is_partial) {
        // final result: same parse in both modes (server-task.h:399)
        http_nonstream = threw ? code : "200";                      // server.cpp:64-77
        http_stream    = threw ? "200+sse_error" + code : "200";    // server-context.cpp:4997, 5097-5102
    } else {
        // partial parses happen only for streamed chunks (server-task.cpp:1023); non-stream never sees them.
        // (if the throwing chunk were the very first result it would be a plain 500, server-context.cpp:4967)
        http_nonstream = "na";
        http_stream    = threw ? "200+sse_error" + code : "200";
    }
}

int main(int argc, char ** argv) {
    const std::string root = argc > 1 ? std::string(argv[1]) + "/" : std::string();

    const std::vector<trunc_format> formats = {
        // Qwen3 (Hermes-style JSON tool calls)
        { "qwen3_hermes_json",  "models/templates/Qwen-Qwen3-0.6B.jinja",         json_cases() },
        // Qwen2.5 (Hermes-style JSON; reference, exercised by test-chat.cpp)
        { "qwen25_hermes_json", "models/templates/Qwen-Qwen2.5-7B-Instruct.jinja", json_cases() },
        // Qwen3-Coder (XML tool calls)
        { "qwen3coder_xml",     "models/templates/Qwen3-Coder.jinja",              xml_cases()  },
        // Qwen3.5 / Qwen3.6 family (XML tool calls inside a thinking template)
        { "qwen35_xml",         "models/templates/Qwen3.5-4B.jinja",               xml_cases()  },
    };

    int n_control_fail = 0;

    for (const auto & fmt : formats) {
        // Build the parser exactly as the server does for an OAI chat request with tools:
        //   server-common.cpp:1039-1064 (inputs), common_chat_templates_apply, then server-schema.cpp:298-327
        //   (format, reasoning_format, generation_prompt, parser.load(chat_parser)).
        common_chat_parser_params pp;
        std::string               chat_format_name = "unknown";
        try {
            common_chat_templates_ptr tmpls(common_chat_templates_init(/* model= */ nullptr,
                                                                       read_file(root + fmt.template_path)));

            common_chat_msg user;
            user.role    = "user";
            user.content = "What is the weather in Paris?";

            common_chat_templates_inputs inputs;
            inputs.messages              = { user };
            inputs.tools                 = { weather_tool };
            inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_AUTO;  // server default (server-common.cpp:915)
            inputs.parallel_tool_calls   = common_chat_templates_get_caps(tmpls.get())["supports_parallel_tool_calls"];  // server-common.cpp:1044
            inputs.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;  // CLI default (common/common.h:656)
            inputs.enable_thinking       = false;  // production Qwen3.x roles run with thinking off
            inputs.add_generation_prompt = true;

            common_chat_params cp = common_chat_templates_apply(tmpls.get(), inputs);

            pp                      = common_chat_parser_params(cp);
            pp.reasoning_format     = inputs.reasoning_format;
            pp.reasoning_in_content = false;  // only stream && deepseek-legacy sets it (server-schema.cpp:310)
            pp.parser.load(cp.parser);
            chat_format_name = common_chat_format_name(cp.format);
        } catch (const std::exception & e) {
            printf("SETUP format=%s template=%s outcome=failed exception=\"%s\"\n", fmt.label.c_str(),
                   fmt.template_path.c_str(), escape_field(e.what(), 120).c_str());
            ++n_control_fail;  // the control case cannot run for this format
            continue;
        }

        for (const auto & tc : fmt.cases) {
            for (int is_partial = 0; is_partial <= 1; ++is_partial) {
                bool            threw                  = false;
                bool            threw_invalid_argument = false;
                std::string     what;
                common_chat_msg msg;
                try {
                    msg = common_chat_parse(tc.text, is_partial != 0, pp);
                } catch (const std::invalid_argument & e) {
                    threw                  = true;
                    threw_invalid_argument = true;
                    what                   = e.what();
                } catch (const std::exception & e) {
                    threw = true;
                    what  = e.what();
                } catch (...) {
                    threw = true;
                    what  = "unknown exception";
                }

                std::string args_valid_json = "na";
                std::string args_is_object  = "na";
                std::string args_joined;
                std::string names_joined;
                if (!threw && !msg.tool_calls.empty()) {
                    bool all_valid  = true;
                    bool all_object = true;
                    for (size_t i = 0; i < msg.tool_calls.size(); ++i) {
                        const auto & call = msg.tool_calls[i];
                        if (i > 0) {
                            args_joined  += " | ";
                            names_joined += ",";
                        }
                        args_joined  += call.arguments;
                        names_joined += call.name;
                        try {
                            json j = json::parse(call.arguments);
                            if (!j.is_object()) {
                                all_object = false;
                            }
                        } catch (const std::exception &) {
                            all_valid  = false;
                            all_object = false;
                        }
                    }
                    args_valid_json = all_valid ? "1" : "0";
                    args_is_object  = all_object ? "1" : "0";
                }

                std::string http_nonstream;
                std::string http_stream;
                derive_http(is_partial != 0, threw, threw_invalid_argument, http_nonstream, http_stream);

                printf("RESULT format=%s case=%c is_partial=%d outcome=%s n_tool_calls=%zu args_valid_json=%s "
                       "content_len=%zu exception=\"%s\" server_http_nonstream=%s server_http_stream=%s "
                       "args_is_object=%s msg_empty=%d chat_format=\"%s\" desc=%s tool_names=\"%s\" args=\"%s\" "
                       "content=\"%s\"\n",
                       fmt.label.c_str(), tc.id, is_partial, threw ? "threw" : "parsed",
                       threw ? (size_t) 0 : msg.tool_calls.size(), args_valid_json.c_str(),
                       threw ? (size_t) 0 : msg.content.size(), escape_field(what, 120).c_str(),
                       http_nonstream.c_str(), http_stream.c_str(), args_is_object.c_str(),
                       threw ? 0 : (msg.empty() ? 1 : 0), escape_field(chat_format_name, 64).c_str(),
                       tc.desc.c_str(), escape_field(names_joined, 64).c_str(),
                       escape_field(args_joined, 160).c_str(), escape_field(msg.content, 80).c_str());
                fflush(stdout);

                // the ONLY assertion: the control must parse as final with valid JSON arguments
                if (tc.id == 'd' && is_partial == 0) {
                    const bool ok = !threw && msg.tool_calls.size() == 1 && args_valid_json == "1" &&
                                    msg.tool_calls[0].name == "get_weather";
                    if (!ok) {
                        fprintf(stderr, "CONTROL FAILED: format=%s case=d did not parse as one valid tool call\n",
                                fmt.label.c_str());
                        ++n_control_fail;
                    }
                }
            }
        }
    }

    printf("SUMMARY control_failures=%d\n", n_control_fail);
    return n_control_fail == 0 ? 0 : 1;
}
