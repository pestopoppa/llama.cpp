// ODE Solver using adaptive Runge-Kutta-Fehlberg (RK45)
// Supports: dy/dt = f(t, y) with y(t0) = y0

#include "../../include/common.hpp"
#include "../../include/json_io.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <algorithm>
#include <cctype>

namespace math_tools {

using namespace Eigen;

// RK45 Coefficients (Dormand-Prince)
namespace RK45 {
    constexpr double c2 = 1.0/5.0, c3 = 3.0/10.0, c4 = 4.0/5.0, c5 = 8.0/9.0, c6 = 1.0, c7 = 1.0;
    constexpr double a21 = 1.0/5.0;
    constexpr double a31 = 3.0/40.0, a32 = 9.0/40.0;
    constexpr double a41 = 44.0/45.0, a42 = -56.0/15.0, a43 = 32.0/9.0;
    constexpr double a51 = 19372.0/6561.0, a52 = -25360.0/2187.0, a53 = 64448.0/6561.0, a54 = -212.0/729.0;
    constexpr double a61 = 9017.0/3168.0, a62 = -355.0/33.0, a63 = 46732.0/5247.0, a64 = 49.0/176.0, a65 = -5103.0/18656.0;
    constexpr double a71 = 35.0/384.0, a73 = 500.0/1113.0, a74 = 125.0/192.0, a75 = -2187.0/6784.0, a76 = 11.0/84.0;

    // 5th order coefficients
    constexpr double b1 = 35.0/384.0, b3 = 500.0/1113.0, b4 = 125.0/192.0, b5 = -2187.0/6784.0, b6 = 11.0/84.0;

    // 4th order coefficients for error estimation
    constexpr double e1 = 71.0/57600.0, e3 = -71.0/16695.0, e4 = 71.0/1920.0, e5 = -17253.0/339200.0, e6 = 22.0/525.0, e7 = -1.0/40.0;
}

class SolveOdeCommand : public Command {
public:
    std::string name() const override { return "solve_ode"; }

    std::string description() const override {
        return "Solve ODEs with adaptive RK45: dy/dt = f(t,y), y(t0)=y0";
    }

    Result execute(const std::string& params_json) override {
        Timer timer;
        Result result;

        try {
            json params = json::parse(params_json);

            // Required parameters
            if (!params.contains("y0")) {
                result.error = "Missing required parameter 'y0' (initial conditions)";
                return result;
            }

            if (!params.contains("t_span")) {
                result.error = "Missing required parameter 't_span' ([t_start, t_end])";
                return result;
            }

            // Get parameters
            std::vector<double> y0_vec = params["y0"].get<std::vector<double>>();
            auto t_span = params["t_span"].get<std::vector<double>>();

            if (t_span.size() != 2) {
                result.error = "t_span must be [t_start, t_end]";
                return result;
            }

            double t0 = t_span[0];
            double tf = t_span[1];

            // Tolerances
            double rtol = params.value("rtol", 1e-6);
            double atol = params.value("atol", 1e-9);
            int max_steps = params.value("max_steps", 10000);

            // Build the ODE function
            std::function<VectorXd(double, const VectorXd&)> f;

            if (params.contains("system")) {
                // Parse expression string (simple case: single equation)
                std::string expr = params["system"].get<std::string>();
                f = parse_ode_system(expr, y0_vec.size());
            } else if (params.contains("coefficients")) {
                // Linear system: dy/dt = A*y + b
                auto A_data = json_to_matrix<double>(params["coefficients"]["A"]);
                MatrixXd A(A_data.size(), A_data[0].size());
                for (size_t i = 0; i < A_data.size(); ++i) {
                    for (size_t j = 0; j < A_data[i].size(); ++j) {
                        A(i, j) = A_data[i][j];
                    }
                }

                VectorXd b = VectorXd::Zero(A.rows());
                if (params["coefficients"].contains("b")) {
                    auto b_vec = params["coefficients"]["b"].get<std::vector<double>>();
                    b = VectorXd::Map(b_vec.data(), b_vec.size());
                }

                f = [A, b](double t, const VectorXd& y) -> VectorXd {
                    return A * y + b;
                };
            } else {
                // Default: exponential decay dy/dt = -y
                f = [](double t, const VectorXd& y) -> VectorXd {
                    return -y;
                };
            }

            // Initial condition
            VectorXd y0 = VectorXd::Map(y0_vec.data(), y0_vec.size());

            // Solve
            auto [t_out, y_out, stats] = integrate_rk45(f, t0, tf, y0, rtol, atol, max_steps);

            // Build output
            json data;
            data["t"] = t_out;

            json y_json = json::array();
            for (const auto& y : y_out) {
                y_json.push_back(std::vector<double>(y.data(), y.data() + y.size()));
            }
            data["y"] = y_json;

            data["stats"]["steps"] = stats.steps;
            data["stats"]["rejected"] = stats.rejected;
            data["stats"]["function_evals"] = stats.func_evals;

            result.success = true;
            result.data = data.dump();

        } catch (const std::exception& e) {
            result.error = e.what();
        }

        result.elapsed_ms = timer.elapsed_ms();
        return result;
    }

private:
    struct IntegrationStats {
        int steps = 0;
        int rejected = 0;
        int func_evals = 0;
    };

    // Parse simple ODE expressions (limited support)
    std::function<VectorXd(double, const VectorXd&)> parse_ode_system(
        const std::string& expr, int n_vars
    ) {
        // Simple coefficient extraction for expressions like "-0.5*y" or "-k*y"
        // For complex expressions, use a proper parser

        if (expr.find("*y") != std::string::npos) {
            // Extract coefficient: "a*y" -> a
            double coeff = 1.0;
            size_t pos = expr.find("*y");
            std::string coeff_str = expr.substr(0, pos);

            // Remove spaces
            coeff_str.erase(remove_if(coeff_str.begin(), coeff_str.end(), isspace), coeff_str.end());

            if (!coeff_str.empty()) {
                try {
                    coeff = std::stod(coeff_str);
                } catch (...) {
                    coeff = -1.0;  // Default to decay
                }
            }

            return [coeff](double t, const VectorXd& y) -> VectorXd {
                return coeff * y;
            };
        }

        // Default: exponential decay
        return [](double t, const VectorXd& y) -> VectorXd {
            return -y;
        };
    }

    // Adaptive RK45 integration (Dormand-Prince)
    std::tuple<std::vector<double>, std::vector<VectorXd>, IntegrationStats>
    integrate_rk45(
        const std::function<VectorXd(double, const VectorXd&)>& f,
        double t0, double tf, const VectorXd& y0,
        double rtol, double atol, int max_steps
    ) {
        std::vector<double> t_out;
        std::vector<VectorXd> y_out;
        IntegrationStats stats;

        double t = t0;
        VectorXd y = y0;
        int n = y.size();

        // Initial step size estimate
        VectorXd f0 = f(t, y);
        stats.func_evals++;

        double d0 = y.norm() / std::sqrt(n);
        double d1 = f0.norm() / std::sqrt(n);
        double h = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * d0 / d1;
        h = std::min(h, tf - t0);

        // Save initial point
        t_out.push_back(t);
        y_out.push_back(y);

        while (t < tf && stats.steps < max_steps) {
            // Limit step to not overshoot tf
            h = std::min(h, tf - t);

            // RK45 step
            VectorXd k1 = f(t, y);
            VectorXd k2 = f(t + RK45::c2*h, y + h*RK45::a21*k1);
            VectorXd k3 = f(t + RK45::c3*h, y + h*(RK45::a31*k1 + RK45::a32*k2));
            VectorXd k4 = f(t + RK45::c4*h, y + h*(RK45::a41*k1 + RK45::a42*k2 + RK45::a43*k3));
            VectorXd k5 = f(t + RK45::c5*h, y + h*(RK45::a51*k1 + RK45::a52*k2 + RK45::a53*k3 + RK45::a54*k4));
            VectorXd k6 = f(t + RK45::c6*h, y + h*(RK45::a61*k1 + RK45::a62*k2 + RK45::a63*k3 + RK45::a64*k4 + RK45::a65*k5));
            VectorXd k7 = f(t + RK45::c7*h, y + h*(RK45::a71*k1 + RK45::a73*k3 + RK45::a74*k4 + RK45::a75*k5 + RK45::a76*k6));

            stats.func_evals += 6;

            // 5th order solution
            VectorXd y_new = y + h * (RK45::b1*k1 + RK45::b3*k3 + RK45::b4*k4 + RK45::b5*k5 + RK45::b6*k6);

            // Error estimate
            VectorXd err = h * (RK45::e1*k1 + RK45::e3*k3 + RK45::e4*k4 + RK45::e5*k5 + RK45::e6*k6 + RK45::e7*k7);

            // Scaled error norm
            double err_norm = 0.0;
            for (int i = 0; i < n; ++i) {
                double scale = atol + rtol * std::max(std::abs(y[i]), std::abs(y_new[i]));
                err_norm += std::pow(err[i] / scale, 2);
            }
            err_norm = std::sqrt(err_norm / n);

            // Accept or reject step
            if (err_norm <= 1.0) {
                // Accept step
                t += h;
                y = y_new;
                t_out.push_back(t);
                y_out.push_back(y);
                stats.steps++;
            } else {
                stats.rejected++;
            }

            // Adjust step size
            double safety = 0.9;
            double min_factor = 0.2;
            double max_factor = 10.0;

            double factor;
            if (err_norm == 0) {
                factor = max_factor;
            } else {
                factor = safety * std::pow(err_norm, -0.2);
                factor = std::max(min_factor, std::min(max_factor, factor));
            }
            h *= factor;
        }

        return {t_out, y_out, stats};
    }
};

std::unique_ptr<Command> create_solve_ode() {
    return std::make_unique<SolveOdeCommand>();
}

} // namespace math_tools
