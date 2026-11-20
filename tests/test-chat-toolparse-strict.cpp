//  PFXC-TOOLPARSE: strict Qwen XML tool-call arguments.
//  Qwen3.6-35B-A3B (C14 GPU, --jinja, GGUF template) sometimes emits nested malformed markup inside an argument value, e.g.
//    <parameter=content>hello</parameter></function><parameter=filePath>\n/a.txt\n</parameter>
//  and `until("\n</parameter>")` copied `hello</parameter></function><parameter=filePath>\n/a.txt` into `content`.
//  The parser must split it when the split is unambiguous, otherwise throw (the server returns an error the client retries).
//  No model, no network; run from the repository root (templates in models/templates) or pass the root as argv[1].

#include "chat.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

static std::string read_file(const std::string & path) {
    std::ifstream fs(path, std::ios_base::binary);
    if (!fs.is_open()) throw std::runtime_error("cannot open template: " + path);
    std::stringstream ss;
    ss << fs.rdbuf();
    return ss.str();
}

static int n_fail = 0;
#define CHECK(cond, what) do { if (!(cond)) { printf("FAIL %s: %s\n", what, #cond); ++n_fail; } else { printf("ok   %s\n", what); } } while (0)

static common_chat_parser_params make_params(const std::string & root, const std::string & tmpl_path) {
    common_chat_templates_ptr tmpls(common_chat_templates_init(nullptr, read_file(root + tmpl_path)));
    common_chat_tool write_tool;
    write_tool.name        = "write";
    write_tool.description = "Write a file";
    write_tool.parameters  = R"({"type":"object","properties":{"content":{"type":"string"},"filePath":{"type":"string"}},"required":["content","filePath"]})";
    common_chat_msg user;
    user.role    = "user";
    user.content = "write it";
    common_chat_templates_inputs inputs;
    inputs.messages              = { user };
    inputs.tools                 = { write_tool };
    inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_AUTO;
    inputs.parallel_tool_calls   = common_chat_templates_get_caps(tmpls.get())["supports_parallel_tool_calls"];
    inputs.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;
    inputs.enable_thinking       = false;
    inputs.add_generation_prompt = true;
    common_chat_params cp = common_chat_templates_apply(tmpls.get(), inputs);
    common_chat_parser_params pp(cp);
    pp.reasoning_format     = inputs.reasoning_format;
    pp.reasoning_in_content = false;
    pp.parser.load(cp.parser);
    return pp;
}

static bool parse(const std::string & text, const common_chat_parser_params & pp, json & args, std::string & err) {
    try {
        common_chat_msg m = common_chat_parse(text, false, pp);
        if (m.tool_calls.size() != 1) { err = "tool_calls=" + std::to_string(m.tool_calls.size()); return false; }
        args = json::parse(m.tool_calls[0].arguments);
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

static common_chat_msg msg_with_args(const json & a) {
    common_chat_msg m;
    m.role = "assistant";
    common_chat_tool_call tc;
    tc.name      = "write";
    tc.arguments = a.dump();
    m.tool_calls.push_back(tc);
    return m;
}

static bool validator_throws(const json & a) {
    auto m = msg_with_args(a);
    try { common_chat_validate_tool_markup(m); } catch (const std::runtime_error &) { return true; }
    return false;
}

int main(int argc, char ** argv) {
    const std::string root = argc > 1 ? std::string(argv[1]) + "/" : std::string();

    // ---- validator unit cases (format independent) ----
    {
        auto m = msg_with_args({{"content", "hello</parameter></function><parameter=filePath>\n/a.txt"}});
        common_chat_validate_tool_markup(m);
        json a = json::parse(m.tool_calls[0].arguments);
        CHECK(a.value("content", "") == "hello" && a.value("filePath", "") == "/a.txt", "validator splits the malformed sample");
    }
    {
        auto m = msg_with_args({{"content", "plain text with <b>html</b>"}, {"filePath", "/x"}});
        const std::string before = m.tool_calls[0].arguments;
        common_chat_validate_tool_markup(m);
        CHECK(m.tool_calls[0].arguments == before, "validator leaves clean values unchanged");
    }
    CHECK(validator_throws({{"content", "a<tool_call>b"}}), "validator rejects <tool_call> inside a value");
    CHECK(validator_throws({{"content", "a</function>b"}}), "validator rejects a stray </function> with no recoverable argument");
    CHECK(validator_throws({{"content", "a</parameter><parameter=content>b"}}), "validator rejects a split that repeats an argument");
    CHECK(validator_throws({{"content", "a</parameter><parameter=filePath>/x"}, {"filePath", "/y"}}), "validator rejects a split that collides with a later argument");
    CHECK(validator_throws({{"content", "a<parameter=filePath>/x"}}), "validator rejects <parameter= without a closing </parameter> for the head");

    // ---- end-to-end through the Qwen3.5/3.6 XML parser (the served path) ----
    // Qwen3.6-35B-A3B.jinja = tokenizer.chat_template of the served Qwen3.6-35B-A3B-MTP-Q8_0.gguf
    for (const std::string tmpl : { "models/templates/Qwen3.6-35B-A3B.jinja", "models/templates/Qwen3.5-4B.jinja", "models/templates/Qwen3-Coder.jinja" }) {
        common_chat_parser_params pp;
        try {
            pp = make_params(root, tmpl);
        } catch (const std::exception & e) {
            printf("FAIL setup %s: %s\n", tmpl.c_str(), e.what());
            ++n_fail;
            continue;
        }
        json a; std::string err;
        const std::string ok_call = "<tool_call>\n<function=write>\n<parameter=content>\nhello\n</parameter>\n<parameter=filePath>\n/a.txt\n</parameter>\n</function>\n</tool_call>";
        bool r = parse(ok_call, pp, a, err);
        CHECK(r && a.value("content", "") == "hello" && a.value("filePath", "") == "/a.txt", (tmpl + ": well-formed call").c_str());

        const std::string bad = "<tool_call>\n<function=write>\n<parameter=content>\nhello</parameter></function><parameter=filePath>\n/a.txt\n</parameter>\n</function>\n</tool_call>";
        r = parse(bad, pp, a, err);
        if (!r) {
            printf("info %s: malformed sample rejected: %s\n", tmpl.c_str(), err.c_str());
            CHECK(err.find("malformed") != std::string::npos || err.find("does not match") != std::string::npos, (tmpl + ": malformed sample fails with a parse error").c_str());
        } else {
            printf("info %s: malformed sample -> %s\n", tmpl.c_str(), a.dump().c_str());
            CHECK(a.value("content", "") == "hello" && a.value("filePath", "") == "/a.txt", (tmpl + ": malformed sample split correctly").c_str());
        }
        bool clean = true;
        if (r) for (auto & [k, v] : a.items()) if (v.is_string() && (v.get<std::string>().find("<parameter=") != std::string::npos || v.get<std::string>().find("</function>") != std::string::npos)) clean = false;
        CHECK(clean, (tmpl + ": no tool markup passes through into a value").c_str());

        // variants of the observed sample: every one must split cleanly or fail, never leak markup
        for (const std::string v : {
                 std::string("<tool_call>\n<function=write>\n<parameter=content>\nhello</parameter>\n</function>\n<parameter=filePath>\n/a.txt\n</parameter>\n</function>\n</tool_call>"),
                 std::string("<tool_call>\n<function=write>\n<parameter=content>\nhello</parameter></function><parameter=filePath>/a.txt</parameter></function>\n</tool_call>"),
                 std::string("<tool_call>\n<function=write>\n<parameter=filePath>\n/a.txt\n</parameter>\n<parameter=content>\nhello</parameter></function><parameter=filePath>\n/b\n</parameter>\n</function>\n</tool_call>") }) {
            json va; std::string verr;
            bool vr = parse(v, pp, va, verr);
            bool vclean = true;
            if (vr) for (auto & [k, x] : va.items()) if (x.is_string() && (x.get<std::string>().find("<parameter=") != std::string::npos || x.get<std::string>().find("</function>") != std::string::npos || x.get<std::string>().find("</parameter>") != std::string::npos)) vclean = false;
            printf("info %s: variant -> %s\n", tmpl.c_str(), vr ? va.dump().c_str() : verr.c_str());
            CHECK(vclean, (tmpl + ": variant leaks no markup").c_str());
            CHECK(vr || verr.find("malformed") != std::string::npos || verr.find("does not match") != std::string::npos,
                  (tmpl + ": variant fails with a server parse error, not garbage").c_str());
        }

        const std::string bad2 = "<tool_call>\n<function=write>\n<parameter=content>\nx<tool_call>y\n</parameter>\n<parameter=filePath>\n/a\n</parameter>\n</function>\n</tool_call>";
        r = parse(bad2, pp, a, err);
        CHECK(!r, (tmpl + ": nested <tool_call> inside a value is rejected").c_str());
    }

    printf("%s (%d failures)\n", n_fail ? "FAILED" : "PASSED", n_fail);
    return n_fail ? 1 : 0;
}
