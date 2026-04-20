// Numerical optimization: minimize/maximize functions
// Supports: BFGS, Nelder-Mead, gradient descent

#include "../../include/common.hpp"
#include "../../include/json_io.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <functional>

namespace math_tools {

using namespace Eigen;

class OptimizeCommand : public Command {
public:
    std::string name() const override { return "optimize"; }

    std::string description() const override {
        return "Numerical optimization: minimize polynomial/quadratic functions";
    }

    Result execute(const std::string& params_json) override {
        Timer timer;
        Result result;

        try {
            json params = json::parse(params_json);

            std::string method = params.value("method", "nelder_mead");
            std::string mode = params.value("mode", "minimize");

            if (!params.contains("x0")) {
                result.error = "Missing required parameter 'x0' (initial guess)";
                return result;
            }

            std::vector<double> x0_vec = params["x0"].get<std::vector<double>>();
            VectorXd x0 = VectorXd::Map(x0_vec.data(), x0_vec.size());

            // Build objective function
            std::function<double(const VectorXd&)> f;
            bool negate = (mode == "maximize");

            if (params.contains("quadratic")) {
                // Quadratic: f(x) = 0.5 * x'Ax + b'x + c
                auto Q_data = json_to_matrix<double>(params["quadratic"]["A"]);
                MatrixXd A(Q_data.size(), Q_data[0].size());
                for (size_t i = 0; i < Q_data.size(); ++i) {
                    for (size_t j = 0; j < Q_data[i].size(); ++j) {
                        A(i, j) = Q_data[i][j];
                    }
                }

                VectorXd b = VectorXd::Zero(A.rows());
                if (params["quadratic"].contains("b")) {
                    auto b_vec = params["quadratic"]["b"].get<std::vector<double>>();
                    b = VectorXd::Map(b_vec.data(), b_vec.size());
                }

                double c = params["quadratic"].value("c", 0.0);

                f = [A, b, c, negate](const VectorXd& x) -> double {
                    double val = 0.5 * x.transpose() * A * x + b.dot(x) + c;
                    return negate ? -val : val;
                };

            } else if (params.contains("rosenbrock")) {
                // Rosenbrock function (n-dimensional) - classic test function
                f = [negate](const VectorXd& x) -> double {
                    double val = 0.0;
                    for (int i = 0; i < x.size() - 1; ++i) {
                        val += 100 * std::pow(x[i+1] - x[i]*x[i], 2) + std::pow(1 - x[i], 2);
                    }
                    return negate ? -val : val;
                };

            } else if (params.contains("sphere")) {
                // Sphere function: f(x) = sum(x_i^2)
                f = [negate](const VectorXd& x) -> double {
                    double val = x.squaredNorm();
                    return negate ? -val : val;
                };

            } else {
                result.error = "No objective function specified. Use 'quadratic', 'rosenbrock', or 'sphere'";
                return result;
            }

            // Optimization parameters
            int max_iter = params.value("max_iter", 1000);
            double tol = params.value("tolerance", 1e-8);

            // Run optimization
            json data;

            if (method == "nelder_mead" || method == "simplex") {
                auto [x_opt, f_opt, iterations, converged] = nelder_mead(f, x0, tol, max_iter);
                data["x"] = std::vector<double>(x_opt.data(), x_opt.data() + x_opt.size());
                data["f"] = negate ? -f_opt : f_opt;
                data["iterations"] = iterations;
                data["converged"] = converged;
                data["method"] = "nelder_mead";

            } else if (method == "gradient_descent" || method == "gd") {
                double lr = params.value("learning_rate", 0.01);
                auto [x_opt, f_opt, iterations, converged] = gradient_descent(f, x0, lr, tol, max_iter);
                data["x"] = std::vector<double>(x_opt.data(), x_opt.data() + x_opt.size());
                data["f"] = negate ? -f_opt : f_opt;
                data["iterations"] = iterations;
                data["converged"] = converged;
                data["method"] = "gradient_descent";

            } else {
                result.error = "Unknown method: " + method + ". Use 'nelder_mead' or 'gradient_descent'";
                return result;
            }

            result.success = true;
            result.data = data.dump();

        } catch (const std::exception& e) {
            result.error = e.what();
        }

        result.elapsed_ms = timer.elapsed_ms();
        return result;
    }

private:
    // Nelder-Mead simplex algorithm
    std::tuple<VectorXd, double, int, bool> nelder_mead(
        const std::function<double(const VectorXd&)>& f,
        const VectorXd& x0,
        double tol,
        int max_iter
    ) {
        int n = x0.size();
        double alpha = 1.0;  // reflection
        double gamma = 2.0;  // expansion
        double rho = 0.5;    // contraction
        double sigma = 0.5;  // shrink

        // Initialize simplex
        std::vector<VectorXd> simplex(n + 1);
        std::vector<double> f_vals(n + 1);

        simplex[0] = x0;
        f_vals[0] = f(x0);

        for (int i = 0; i < n; ++i) {
            VectorXd xi = x0;
            xi[i] += (xi[i] != 0) ? 0.05 * xi[i] : 0.00025;
            simplex[i + 1] = xi;
            f_vals[i + 1] = f(xi);
        }

        int iterations = 0;
        bool converged = false;

        while (iterations < max_iter) {
            // Sort by function value
            std::vector<int> idx(n + 1);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int i, int j) {
                return f_vals[i] < f_vals[j];
            });

            std::vector<VectorXd> sorted_simplex(n + 1);
            std::vector<double> sorted_f(n + 1);
            for (int i = 0; i <= n; ++i) {
                sorted_simplex[i] = simplex[idx[i]];
                sorted_f[i] = f_vals[idx[i]];
            }
            simplex = sorted_simplex;
            f_vals = sorted_f;

            // Check convergence
            double range = f_vals[n] - f_vals[0];
            if (range < tol) {
                converged = true;
                break;
            }

            // Compute centroid (excluding worst point)
            VectorXd centroid = VectorXd::Zero(n);
            for (int i = 0; i < n; ++i) {
                centroid += simplex[i];
            }
            centroid /= n;

            // Reflection
            VectorXd xr = centroid + alpha * (centroid - simplex[n]);
            double fr = f(xr);

            if (fr >= f_vals[0] && fr < f_vals[n-1]) {
                simplex[n] = xr;
                f_vals[n] = fr;
            }
            else if (fr < f_vals[0]) {
                // Expansion
                VectorXd xe = centroid + gamma * (xr - centroid);
                double fe = f(xe);

                if (fe < fr) {
                    simplex[n] = xe;
                    f_vals[n] = fe;
                } else {
                    simplex[n] = xr;
                    f_vals[n] = fr;
                }
            }
            else {
                // Contraction
                VectorXd xc;
                if (fr < f_vals[n]) {
                    xc = centroid + rho * (xr - centroid);
                } else {
                    xc = centroid + rho * (simplex[n] - centroid);
                }
                double fc = f(xc);

                if (fc < std::min(fr, f_vals[n])) {
                    simplex[n] = xc;
                    f_vals[n] = fc;
                } else {
                    // Shrink
                    for (int i = 1; i <= n; ++i) {
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0]);
                        f_vals[i] = f(simplex[i]);
                    }
                }
            }

            iterations++;
        }

        return {simplex[0], f_vals[0], iterations, converged};
    }

    // Gradient descent with numerical gradient
    std::tuple<VectorXd, double, int, bool> gradient_descent(
        const std::function<double(const VectorXd&)>& f,
        const VectorXd& x0,
        double lr,
        double tol,
        int max_iter
    ) {
        int n = x0.size();
        VectorXd x = x0;
        double fx = f(x);
        int iterations = 0;
        bool converged = false;

        double eps = 1e-8;

        while (iterations < max_iter) {
            // Numerical gradient
            VectorXd grad(n);
            for (int i = 0; i < n; ++i) {
                VectorXd x_plus = x;
                VectorXd x_minus = x;
                x_plus[i] += eps;
                x_minus[i] -= eps;
                grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps);
            }

            // Check convergence
            if (grad.norm() < tol) {
                converged = true;
                break;
            }

            // Update
            x -= lr * grad;
            double fx_new = f(x);

            // Simple line search backtracking
            while (fx_new > fx && lr > 1e-10) {
                lr *= 0.5;
                x += lr * grad;  // Undo too-large step
                x -= lr * grad;
                fx_new = f(x);
            }

            fx = fx_new;
            iterations++;
        }

        return {x, fx, iterations, converged};
    }
};

std::unique_ptr<Command> create_optimize() {
    return std::make_unique<OptimizeCommand>();
}

} // namespace math_tools
