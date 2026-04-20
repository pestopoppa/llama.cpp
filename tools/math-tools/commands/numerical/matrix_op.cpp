// Matrix operations using Eigen
// Supports: solve, inverse, determinant, eigenvalues, SVD, LU, QR

#include "../../include/common.hpp"
#include "../../include/json_io.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace math_tools {

using namespace Eigen;

class MatrixOpCommand : public Command {
public:
    std::string name() const override { return "matrix_op"; }

    std::string description() const override {
        return "Linear algebra: solve, inverse, det, eigen, svd, lu, qr";
    }

    Result execute(const std::string& params_json) override {
        Timer timer;
        Result result;

        try {
            json params = json::parse(params_json);
            std::string operation = params.value("operation", "solve");

            if (operation == "solve") {
                result = solve_system(params);
            } else if (operation == "inverse") {
                result = matrix_inverse(params);
            } else if (operation == "det") {
                result = determinant(params);
            } else if (operation == "eigen") {
                result = eigenvalues(params);
            } else if (operation == "svd") {
                result = singular_values(params);
            } else if (operation == "lu") {
                result = lu_decomposition(params);
            } else if (operation == "qr") {
                result = qr_decomposition(params);
            } else if (operation == "rank") {
                result = matrix_rank(params);
            } else {
                result.error = "Unknown operation: " + operation;
            }
        } catch (const std::exception& e) {
            result.error = e.what();
        }

        result.elapsed_ms = timer.elapsed_ms();
        return result;
    }

private:
    // Convert JSON array to Eigen matrix
    MatrixXd json_to_eigen(const json& j) {
        auto mat = json_to_matrix<double>(j);
        int rows = mat.size();
        int cols = mat[0].size();

        MatrixXd eigen_mat(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                eigen_mat(i, j) = mat[i][j];
            }
        }
        return eigen_mat;
    }

    // Convert Eigen matrix to JSON array
    json eigen_to_json(const MatrixXd& mat) {
        json result = json::array();
        for (int i = 0; i < mat.rows(); ++i) {
            json row = json::array();
            for (int j = 0; j < mat.cols(); ++j) {
                row.push_back(mat(i, j));
            }
            result.push_back(row);
        }
        return result;
    }

    // Solve Ax = b
    Result solve_system(const json& params) {
        Result result;

        if (!params.contains("A") || !params.contains("b")) {
            result.error = "solve requires 'A' (matrix) and 'b' (vector)";
            return result;
        }

        MatrixXd A = json_to_eigen(params["A"]);
        std::vector<double> b_vec = params["b"].get<std::vector<double>>();
        VectorXd b = VectorXd::Map(b_vec.data(), b_vec.size());

        // Use ColPivHouseholderQR for numerical stability
        VectorXd x = A.colPivHouseholderQr().solve(b);

        // Check solution quality
        double relative_error = (A * x - b).norm() / b.norm();

        json data;
        data["x"] = std::vector<double>(x.data(), x.data() + x.size());
        data["relative_error"] = relative_error;

        result.success = true;
        result.data = data.dump();
        return result;
    }

    // Matrix inverse
    Result matrix_inverse(const json& params) {
        Result result;

        if (!params.contains("A")) {
            result.error = "inverse requires 'A' (matrix)";
            return result;
        }

        MatrixXd A = json_to_eigen(params["A"]);

        if (A.rows() != A.cols()) {
            result.error = "Matrix must be square for inverse";
            return result;
        }

        FullPivLU<MatrixXd> lu(A);
        if (!lu.isInvertible()) {
            result.error = "Matrix is singular (not invertible)";
            return result;
        }

        MatrixXd inv = A.inverse();

        json data;
        data["inverse"] = eigen_to_json(inv);
        data["condition_number"] = A.norm() * inv.norm();

        result.success = true;
        result.data = data.dump();
        return result;
    }

    // Determinant
    Result determinant(const json& params) {
        Result result;

        if (!params.contains("A")) {
            result.error = "det requires 'A' (matrix)";
            return result;
        }

        MatrixXd A = json_to_eigen(params["A"]);

        if (A.rows() != A.cols()) {
            result.error = "Matrix must be square for determinant";
            return result;
        }

        json data;
        data["determinant"] = A.determinant();

        result.success = true;
        result.data = data.dump();
        return result;
    }

    // Eigenvalues and eigenvectors
    Result eigenvalues(const json& params) {
        Result result;

        if (!params.contains("A")) {
            result.error = "eigen requires 'A' (matrix)";
            return result;
        }

        MatrixXd A = json_to_eigen(params["A"]);

        if (A.rows() != A.cols()) {
            result.error = "Matrix must be square for eigenvalues";
            return result;
        }

        bool vectors = params.value("vectors", false);

        EigenSolver<MatrixXd> solver(A);

        json data;

        // Eigenvalues (may be complex)
        auto evals = solver.eigenvalues();
        json eigenvalues_json = json::array();
        for (int i = 0; i < evals.size(); ++i) {
            if (std::abs(evals[i].imag()) < 1e-10) {
                eigenvalues_json.push_back(evals[i].real());
            } else {
                json complex_val;
                complex_val["real"] = evals[i].real();
                complex_val["imag"] = evals[i].imag();
                eigenvalues_json.push_back(complex_val);
            }
        }
        data["eigenvalues"] = eigenvalues_json;

        // Optionally include eigenvectors
        if (vectors) {
            auto evecs = solver.eigenvectors();
            json eigenvectors_json = json::array();
            for (int j = 0; j < evecs.cols(); ++j) {
                json vec = json::array();
                for (int i = 0; i < evecs.rows(); ++i) {
                    if (std::abs(evecs(i, j).imag()) < 1e-10) {
                        vec.push_back(evecs(i, j).real());
                    } else {
                        json complex_val;
                        complex_val["real"] = evecs(i, j).real();
                        complex_val["imag"] = evecs(i, j).imag();
                        vec.push_back(complex_val);
                    }
                }
                eigenvectors_json.push_back(vec);
            }
            data["eigenvectors"] = eigenvectors_json;
        }

        result.success = true;
        result.data = data.dump();
        return result;
    }

    // Singular Value Decomposition
    Result singular_values(const json& params) {
        Result result;

        if (!params.contains("A")) {
            result.error = "svd requires 'A' (matrix)";
            return result;
        }

        MatrixXd A = json_to_eigen(params["A"]);
        bool full = params.value("full", false);

        JacobiSVD<MatrixXd> svd(A, full ? ComputeFullU | ComputeFullV : ComputeThinU | ComputeThinV);

        json data;
        data["singular_values"] = std::vector<double>(
            svd.singularValues().data(),
            svd.singularValues().data() + svd.singularValues().size()
        );

        if (full || params.value("matrices", false)) {
            data["U"] = eigen_to_json(svd.matrixU());
            data["V"] = eigen_to_json(svd.matrixV());
        }

        data["rank"] = svd.rank();

        result.success = true;
        result.data = data.dump();
        return result;
    }

    // LU Decomposition
    Result lu_decomposition(const json& params) {
        Result result;

        if (!params.contains("A")) {
            result.error = "lu requires 'A' (matrix)";
            return result;
        }

        MatrixXd A = json_to_eigen(params["A"]);
        PartialPivLU<MatrixXd> lu(A);

        json data;
        data["L"] = eigen_to_json(MatrixXd(lu.matrixLU().triangularView<StrictlyLower>()) + MatrixXd::Identity(A.rows(), A.cols()));
        data["U"] = eigen_to_json(MatrixXd(lu.matrixLU().triangularView<Upper>()));

        // Permutation as indices
        auto perm = lu.permutationP().indices();
        data["P"] = std::vector<int>(perm.data(), perm.data() + perm.size());

        result.success = true;
        result.data = data.dump();
        return result;
    }

    // QR Decomposition
    Result qr_decomposition(const json& params) {
        Result result;

        if (!params.contains("A")) {
            result.error = "qr requires 'A' (matrix)";
            return result;
        }

        MatrixXd A = json_to_eigen(params["A"]);
        HouseholderQR<MatrixXd> qr(A);

        json data;
        data["Q"] = eigen_to_json(qr.householderQ());
        data["R"] = eigen_to_json(qr.matrixQR().triangularView<Upper>());

        result.success = true;
        result.data = data.dump();
        return result;
    }

    // Matrix rank
    Result matrix_rank(const json& params) {
        Result result;

        if (!params.contains("A")) {
            result.error = "rank requires 'A' (matrix)";
            return result;
        }

        MatrixXd A = json_to_eigen(params["A"]);
        double tol = params.value("tolerance", 1e-10);

        FullPivLU<MatrixXd> lu(A);
        lu.setThreshold(tol);

        json data;
        data["rank"] = lu.rank();
        data["tolerance"] = tol;

        result.success = true;
        result.data = data.dump();
        return result;
    }
};

std::unique_ptr<Command> create_matrix_op() {
    return std::make_unique<MatrixOpCommand>();
}

} // namespace math_tools
