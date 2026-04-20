// llama-math-tools: Native computational tools for LLM orchestration
//
// Philosophy: LLMs reason. Tools execute.
// Every tool call should save 100-5000 tokens of LLM generation.
//
// Usage:
//   echo '{"command":"matrix_op","operation":"solve","A":[[3,1],[1,2]],"b":[9,8]}' | ./llama-math-tools
//   echo '{"command":"help"}' | ./llama-math-tools

#include "include/common.hpp"
#include "include/json_io.hpp"
#include <iostream>
#include <memory>
#include <map>
#include <functional>

// Forward declarations of command factories
namespace math_tools {
    // Numerical
    std::unique_ptr<Command> create_matrix_op();
    std::unique_ptr<Command> create_solve_ode();
    std::unique_ptr<Command> create_optimize();
    // Statistical
    std::unique_ptr<Command> create_monte_carlo();
    std::unique_ptr<Command> create_mcmc();
    std::unique_ptr<Command> create_bayesopt();
    // Visualization
    std::unique_ptr<Command> create_plot_braille();
    std::unique_ptr<Command> create_render_math();
    std::unique_ptr<Command> create_plot_sixel();
}

using namespace math_tools;

// Command registry
class CommandRegistry {
public:
    using Factory = std::function<std::unique_ptr<Command>()>;

    void register_command(const std::string& name, Factory factory) {
        factories_[name] = std::move(factory);
    }

    std::unique_ptr<Command> create(const std::string& name) const {
        auto it = factories_.find(name);
        if (it == factories_.end()) {
            return nullptr;
        }
        return it->second();
    }

    std::vector<std::string> list() const {
        std::vector<std::string> names;
        for (const auto& [name, _] : factories_) {
            names.push_back(name);
        }
        return names;
    }

private:
    std::map<std::string, Factory> factories_;
};

// Build help JSON
json build_help(const CommandRegistry& registry) {
    json help;
    help["commands"] = json::array();

    for (const auto& name : registry.list()) {
        auto cmd = registry.create(name);
        if (cmd) {
            json cmd_info;
            cmd_info["name"] = cmd->name();
            cmd_info["description"] = cmd->description();
            help["commands"].push_back(cmd_info);
        }
    }

    return help;
}

int main(int argc, char* argv[]) {
    // Register all commands
    CommandRegistry registry;
    // Numerical
    registry.register_command("matrix_op", create_matrix_op);
    registry.register_command("solve_ode", create_solve_ode);
    registry.register_command("optimize", create_optimize);
    // Statistical
    registry.register_command("monte_carlo", create_monte_carlo);
    registry.register_command("mcmc", create_mcmc);
    registry.register_command("bayesopt", create_bayesopt);
    // Visualization
    registry.register_command("plot", create_plot_braille);
    registry.register_command("render_math", create_render_math);
    registry.register_command("plot_sixel", create_plot_sixel);

    // Handle --help flag
    if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << "llama-math-tools: Native computational tools for LLM orchestration\n\n";
        std::cout << "Usage: echo '{\"command\":\"NAME\", ...}' | llama-math-tools\n\n";
        std::cout << "Available commands:\n";
        for (const auto& name : registry.list()) {
            auto cmd = registry.create(name);
            if (cmd) {
                std::cout << "  " << cmd->name() << " - " << cmd->description() << "\n";
            }
        }
        std::cout << "\nSend JSON to stdin, receive JSON from stdout.\n";
        return 0;
    }

    try {
        // Read input JSON
        json input = read_input();

        // Get command name
        if (!input.contains("command")) {
            write_error("Missing 'command' field");
            return 1;
        }

        std::string command_name = input["command"].get<std::string>();

        // Handle 'help' command
        if (command_name == "help") {
            Result result;
            result.success = true;
            result.data = build_help(registry).dump();
            write_output(result);
            return 0;
        }

        // Create and execute command
        auto command = registry.create(command_name);
        if (!command) {
            write_error("Unknown command: " + command_name);
            return 1;
        }

        // Execute with parameters (everything except 'command' key)
        json params = input;
        params.erase("command");

        Result result = command->execute(params.dump());
        write_output(result);

        return result.success ? 0 : 1;

    } catch (const json::parse_error& e) {
        write_error(std::string("JSON parse error: ") + e.what());
        return 1;
    } catch (const std::exception& e) {
        write_error(std::string("Error: ") + e.what());
        return 1;
    }

    return 0;
}
