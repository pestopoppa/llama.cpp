/**
 * @file render_math.cpp
 * @brief LaTeX to Unicode/ASCII math renderer
 *
 * Converts LaTeX mathematical expressions to Unicode or ASCII representation.
 *
 * Usage:
 *   {"command": "render_math", "latex": "\\frac{dy}{dx}", "format": "unicode"}
 */

#include "../../include/common.hpp"
#include "../../include/json_io.hpp"
#include <regex>
#include <map>
#include <sstream>

namespace math_tools {

class RenderMathCommand : public Command {
public:
    std::string name() const override { return "render_math"; }
    std::string description() const override {
        return "Render LaTeX to Unicode/ASCII";
    }

    Result execute(const std::string& params_json) override {
        Timer timer;
        Result result;

        try {
            json params = json::parse(params_json);

            if (!params.contains("latex")) {
                result.error = "Missing required parameter: latex";
                return result;
            }

            std::string latex = params["latex"];
            std::string format = params.value("format", "unicode");
            bool use_unicode = (format == "unicode");

            std::string rendered = render(latex, use_unicode);

            json data;
            data["rendered"] = rendered;
            data["original"] = latex;
            data["format"] = format;

            result.success = true;
            result.data = data.dump();

        } catch (const std::exception& e) {
            result.error = std::string("Render error: ") + e.what();
        }

        result.elapsed_ms = timer.elapsed_ms();
        return result;
    }

private:
    // Greek letter mappings
    static const std::map<std::string, std::pair<std::string, std::string>>& greekLetters() {
        static const std::map<std::string, std::pair<std::string, std::string>> m = {
            {"alpha", {"α", "alpha"}},
            {"beta", {"β", "beta"}},
            {"gamma", {"γ", "gamma"}},
            {"delta", {"δ", "delta"}},
            {"epsilon", {"ε", "epsilon"}},
            {"zeta", {"ζ", "zeta"}},
            {"eta", {"η", "eta"}},
            {"theta", {"θ", "theta"}},
            {"iota", {"ι", "iota"}},
            {"kappa", {"κ", "kappa"}},
            {"lambda", {"λ", "lambda"}},
            {"mu", {"μ", "mu"}},
            {"nu", {"ν", "nu"}},
            {"xi", {"ξ", "xi"}},
            {"pi", {"π", "pi"}},
            {"rho", {"ρ", "rho"}},
            {"sigma", {"σ", "sigma"}},
            {"tau", {"τ", "tau"}},
            {"upsilon", {"υ", "upsilon"}},
            {"phi", {"φ", "phi"}},
            {"chi", {"χ", "chi"}},
            {"psi", {"ψ", "psi"}},
            {"omega", {"ω", "omega"}},
            {"Alpha", {"Α", "Alpha"}},
            {"Beta", {"Β", "Beta"}},
            {"Gamma", {"Γ", "Gamma"}},
            {"Delta", {"Δ", "Delta"}},
            {"Theta", {"Θ", "Theta"}},
            {"Lambda", {"Λ", "Lambda"}},
            {"Xi", {"Ξ", "Xi"}},
            {"Pi", {"Π", "Pi"}},
            {"Sigma", {"Σ", "Sigma"}},
            {"Phi", {"Φ", "Phi"}},
            {"Psi", {"Ψ", "Psi"}},
            {"Omega", {"Ω", "Omega"}},
            {"varepsilon", {"ε", "epsilon"}},
            {"vartheta", {"ϑ", "theta"}},
            {"varphi", {"ϕ", "phi"}},
            {"varpi", {"ϖ", "pi"}},
        };
        return m;
    }

    // Symbol mappings
    static const std::map<std::string, std::pair<std::string, std::string>>& symbols() {
        static const std::map<std::string, std::pair<std::string, std::string>> m = {
            {"times", {"×", "*"}},
            {"div", {"÷", "/"}},
            {"cdot", {"·", "."}},
            {"pm", {"±", "+/-"}},
            {"mp", {"∓", "-/+"}},
            {"leq", {"≤", "<="}},
            {"geq", {"≥", ">="}},
            {"neq", {"≠", "!="}},
            {"approx", {"≈", "~="}},
            {"equiv", {"≡", "==="}},
            {"to", {"→", "->"}},
            {"rightarrow", {"→", "->"}},
            {"leftarrow", {"←", "<-"}},
            {"Rightarrow", {"⇒", "=>"}},
            {"Leftarrow", {"⇐", "<="}},
            {"in", {"∈", "in"}},
            {"notin", {"∉", "!in"}},
            {"subset", {"⊂", "subset"}},
            {"cup", {"∪", "U"}},
            {"cap", {"∩", "^"}},
            {"emptyset", {"∅", "{}"}},
            {"forall", {"∀", "forall"}},
            {"exists", {"∃", "exists"}},
            {"neg", {"¬", "!"}},
            {"land", {"∧", "&&"}},
            {"lor", {"∨", "||"}},
            {"infty", {"∞", "inf"}},
            {"partial", {"∂", "d"}},
            {"nabla", {"∇", "nabla"}},
            {"int", {"∫", "integral"}},
            {"sum", {"∑", "sum"}},
            {"prod", {"∏", "prod"}},
            {"sqrt", {"√", "sqrt"}},
            {"prime", {"′", "'"}},
        };
        return m;
    }

    // Superscript mappings
    static const std::map<char, std::string>& superscripts() {
        static const std::map<char, std::string> m = {
            {'0', "⁰"}, {'1', "¹"}, {'2', "²"}, {'3', "³"}, {'4', "⁴"},
            {'5', "⁵"}, {'6', "⁶"}, {'7', "⁷"}, {'8', "⁸"}, {'9', "⁹"},
            {'+', "⁺"}, {'-', "⁻"}, {'n', "ⁿ"}, {'i', "ⁱ"},
        };
        return m;
    }

    // Subscript mappings
    static const std::map<char, std::string>& subscripts() {
        static const std::map<char, std::string> m = {
            {'0', "₀"}, {'1', "₁"}, {'2', "₂"}, {'3', "₃"}, {'4', "₄"},
            {'5', "₅"}, {'6', "₆"}, {'7', "₇"}, {'8', "₈"}, {'9', "₉"},
            {'+', "₊"}, {'-', "₋"}, {'a', "ₐ"}, {'e', "ₑ"}, {'o', "ₒ"},
            {'x', "ₓ"}, {'i', "ᵢ"}, {'n', "ₙ"}, {'m', "ₘ"},
        };
        return m;
    }

    std::string render(const std::string& latex, bool unicode) {
        std::string result = latex;

        // Remove \left and \right
        result = std::regex_replace(result, std::regex("\\\\left"), "");
        result = std::regex_replace(result, std::regex("\\\\right"), "");

        // Handle fractions: \frac{a}{b} -> a/b
        std::regex frac_re("\\\\frac\\{([^{}]*)\\}\\{([^{}]*)\\}");
        result = std::regex_replace(result, frac_re, "$1/$2");
        result = std::regex_replace(result, frac_re, "$1/$2");

        // Handle sqrt: \sqrt{x} -> √x or sqrt(x)
        std::regex sqrt_re("\\\\sqrt\\{([^{}]*)\\}");
        if (unicode) {
            result = std::regex_replace(result, sqrt_re, "√($1)");
        } else {
            result = std::regex_replace(result, sqrt_re, "sqrt($1)");
        }

        // Handle superscripts: x^{2} -> x²
        if (unicode) {
            std::regex sup_re("\\^\\{([^{}]*)\\}");
            std::smatch match;
            while (std::regex_search(result, match, sup_re)) {
                std::string content = match[1];
                std::string sup = toSuperscript(content);
                result = match.prefix().str() + sup + match.suffix().str();
            }

            std::regex sup_single_re("\\^([0-9n])");
            while (std::regex_search(result, match, sup_single_re)) {
                char c = match[1].str()[0];
                auto it = superscripts().find(c);
                std::string sup = (it != superscripts().end()) ? it->second : match[1].str();
                result = match.prefix().str() + sup + match.suffix().str();
            }
        } else {
            result = std::regex_replace(result, std::regex("\\^\\{([^{}]*)\\}"), "^($1)");
        }

        // Handle subscripts: x_{n} -> xₙ
        if (unicode) {
            std::regex sub_re("_\\{([^{}]*)\\}");
            std::smatch match;
            while (std::regex_search(result, match, sub_re)) {
                std::string content = match[1];
                std::string sub = toSubscript(content);
                result = match.prefix().str() + sub + match.suffix().str();
            }

            std::regex sub_single_re("_([0-9])");
            while (std::regex_search(result, match, sub_single_re)) {
                char c = match[1].str()[0];
                auto it = subscripts().find(c);
                std::string sub = (it != subscripts().end()) ? it->second : match[1].str();
                result = match.prefix().str() + sub + match.suffix().str();
            }
        } else {
            result = std::regex_replace(result, std::regex("_\\{([^{}]*)\\}"), "_($1)");
        }

        // Replace Greek letters
        for (const auto& [name, pair] : greekLetters()) {
            std::regex re("\\\\(" + name + ")(?![a-zA-Z])");
            result = std::regex_replace(result, re, unicode ? pair.first : pair.second);
        }

        // Replace symbols
        for (const auto& [name, pair] : symbols()) {
            std::regex re("\\\\(" + name + ")(?![a-zA-Z])");
            result = std::regex_replace(result, re, unicode ? pair.first : pair.second);
        }

        // Clean up remaining braces
        result = std::regex_replace(result, std::regex("\\{([^{}]*)\\}"), "$1");

        // Clean up spaces
        result = std::regex_replace(result, std::regex("\\\\,"), " ");
        result = std::regex_replace(result, std::regex("\\\\;"), " ");
        result = std::regex_replace(result, std::regex("\\\\quad"), "  ");

        // Remove text commands
        result = std::regex_replace(result, std::regex("\\\\text\\{([^{}]*)\\}"), "$1");
        result = std::regex_replace(result, std::regex("\\\\mathrm\\{([^{}]*)\\}"), "$1");

        return result;
    }

    std::string toSuperscript(const std::string& s) {
        std::string result;
        for (char c : s) {
            auto it = superscripts().find(c);
            if (it != superscripts().end()) {
                result += it->second;
            } else {
                result += c;
            }
        }
        return result;
    }

    std::string toSubscript(const std::string& s) {
        std::string result;
        for (char c : s) {
            auto it = subscripts().find(c);
            if (it != subscripts().end()) {
                result += it->second;
            } else {
                result += c;
            }
        }
        return result;
    }
};

// Factory function
std::unique_ptr<Command> create_render_math() {
    return std::make_unique<RenderMathCommand>();
}

} // namespace math_tools
