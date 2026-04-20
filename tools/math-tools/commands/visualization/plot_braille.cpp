// Braille character plotting for terminal visualization
// Each character represents a 2x4 grid of dots
// Resolution: width*2 x height*4 pixels

#include "../../include/common.hpp"
#include "../../include/json_io.hpp"
#include <cmath>
#include <algorithm>
#include <sstream>

namespace math_tools {

class PlotBrailleCommand : public Command {
public:
    std::string name() const override { return "plot"; }

    std::string description() const override {
        return "Plot data as braille characters for terminal display";
    }

    Result execute(const std::string& params_json) override {
        Timer timer;
        Result result;

        try {
            json params = json::parse(params_json);

            std::string plot_type = params.value("type", "scatter");
            int width = params.value("width", 80);
            int height = params.value("height", 20);

            if (plot_type == "scatter" || plot_type == "line") {
                result = plot_xy(params, width, height, plot_type == "line");
            } else if (plot_type == "histogram") {
                result = plot_histogram(params, width, height);
            } else if (plot_type == "function") {
                result = plot_function(params, width, height);
            } else {
                result.error = "Unknown plot type: " + plot_type;
            }

        } catch (const std::exception& e) {
            result.error = e.what();
        }

        result.elapsed_ms = timer.elapsed_ms();
        return result;
    }

private:
    // Braille character encoding
    // Dot positions:
    //   1 4
    //   2 5
    //   3 6
    //   7 8
    static constexpr char32_t BRAILLE_BASE = 0x2800;  // Empty braille pattern

    // Map from (x, y) in 2x4 grid to braille dot bit
    int dot_bit(int x, int y) const {
        static const int bits[2][4] = {
            {0, 1, 2, 6},  // x=0: dots 1,2,3,7
            {3, 4, 5, 7}   // x=1: dots 4,5,6,8
        };
        return 1 << bits[x][y];
    }

    // Canvas for plotting
    class BrailleCanvas {
    public:
        BrailleCanvas(int w, int h) : width_(w), height_(h) {
            pixels_.resize(h * 4, std::vector<bool>(w * 2, false));
        }

        void set_pixel(int x, int y) {
            if (x >= 0 && x < width_ * 2 && y >= 0 && y < height_ * 4) {
                pixels_[y][x] = true;
            }
        }

        void draw_line(int x0, int y0, int x1, int y1) {
            // Bresenham's line algorithm
            int dx = std::abs(x1 - x0);
            int dy = std::abs(y1 - y0);
            int sx = x0 < x1 ? 1 : -1;
            int sy = y0 < y1 ? 1 : -1;
            int err = dx - dy;

            while (true) {
                set_pixel(x0, y0);
                if (x0 == x1 && y0 == y1) break;
                int e2 = 2 * err;
                if (e2 > -dy) {
                    err -= dy;
                    x0 += sx;
                }
                if (e2 < dx) {
                    err += dx;
                    y0 += sy;
                }
            }
        }

        std::string render() const {
            std::stringstream ss;

            for (int cy = 0; cy < height_; ++cy) {
                for (int cx = 0; cx < width_; ++cx) {
                    int pattern = 0;
                    for (int dy = 0; dy < 4; ++dy) {
                        for (int dx = 0; dx < 2; ++dx) {
                            int px = cx * 2 + dx;
                            int py = cy * 4 + dy;
                            if (py < height_ * 4 && pixels_[py][px]) {
                                // Map (dx, dy) to braille dot position
                                static const int bits[2][4] = {
                                    {0, 1, 2, 6},  // x=0
                                    {3, 4, 5, 7}   // x=1
                                };
                                pattern |= (1 << bits[dx][dy]);
                            }
                        }
                    }

                    // Output UTF-8 braille character
                    char32_t ch = 0x2800 + pattern;
                    // Encode as UTF-8
                    if (ch < 0x80) {
                        ss << static_cast<char>(ch);
                    } else if (ch < 0x800) {
                        ss << static_cast<char>(0xC0 | (ch >> 6));
                        ss << static_cast<char>(0x80 | (ch & 0x3F));
                    } else {
                        ss << static_cast<char>(0xE0 | (ch >> 12));
                        ss << static_cast<char>(0x80 | ((ch >> 6) & 0x3F));
                        ss << static_cast<char>(0x80 | (ch & 0x3F));
                    }
                }
                ss << "\n";
            }

            return ss.str();
        }

        int width() const { return width_; }
        int height() const { return height_; }
        int pixel_width() const { return width_ * 2; }
        int pixel_height() const { return height_ * 4; }

    private:
        int width_, height_;
        std::vector<std::vector<bool>> pixels_;
    };

    // Plot x, y data
    Result plot_xy(const json& params, int width, int height, bool line) {
        Result result;

        if (!params.contains("x") || !params.contains("y")) {
            result.error = "scatter/line plot requires 'x' and 'y' arrays";
            return result;
        }

        auto x = params["x"].get<std::vector<double>>();
        auto y = params["y"].get<std::vector<double>>();

        if (x.size() != y.size()) {
            result.error = "x and y arrays must have same length";
            return result;
        }

        // Find bounds
        double x_min = *std::min_element(x.begin(), x.end());
        double x_max = *std::max_element(x.begin(), x.end());
        double y_min = *std::min_element(y.begin(), y.end());
        double y_max = *std::max_element(y.begin(), y.end());

        // Add margin
        double x_margin = (x_max - x_min) * 0.05;
        double y_margin = (y_max - y_min) * 0.05;
        x_min -= x_margin;
        x_max += x_margin;
        y_min -= y_margin;
        y_max += y_margin;

        // Handle constant values
        if (x_max == x_min) {
            x_min -= 1;
            x_max += 1;
        }
        if (y_max == y_min) {
            y_min -= 1;
            y_max += 1;
        }

        BrailleCanvas canvas(width, height);
        int pw = canvas.pixel_width();
        int ph = canvas.pixel_height();

        // Convert data to pixel coordinates
        std::vector<int> px(x.size()), py(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            px[i] = static_cast<int>((x[i] - x_min) / (x_max - x_min) * (pw - 1));
            py[i] = static_cast<int>((y_max - y[i]) / (y_max - y_min) * (ph - 1));  // Flip y
        }

        if (line) {
            // Draw connected lines
            for (size_t i = 1; i < x.size(); ++i) {
                canvas.draw_line(px[i-1], py[i-1], px[i], py[i]);
            }
        } else {
            // Draw points
            for (size_t i = 0; i < x.size(); ++i) {
                canvas.set_pixel(px[i], py[i]);
            }
        }

        // Build output
        std::string plot = canvas.render();

        // Add axis labels
        std::stringstream ss;
        ss << params.value("title", "Plot") << "\n";
        ss << "y: [" << y_min << ", " << y_max << "]\n";
        ss << plot;
        ss << "x: [" << x_min << ", " << x_max << "]";

        json data;
        data["plot"] = ss.str();
        data["format"] = "braille";
        data["width"] = width;
        data["height"] = height;
        data["x_range"] = {x_min, x_max};
        data["y_range"] = {y_min, y_max};
        data["n_points"] = x.size();

        result.success = true;
        result.data = data.dump();
        return result;
    }

    // Plot histogram
    Result plot_histogram(const json& params, int width, int height) {
        Result result;

        if (!params.contains("data")) {
            result.error = "histogram requires 'data' array";
            return result;
        }

        auto data_vec = params["data"].get<std::vector<double>>();
        int n_bins = params.value("bins", 20);

        // Find bounds
        double d_min = *std::min_element(data_vec.begin(), data_vec.end());
        double d_max = *std::max_element(data_vec.begin(), data_vec.end());

        // Compute histogram
        std::vector<int> counts(n_bins, 0);
        double bin_width = (d_max - d_min) / n_bins;

        for (double v : data_vec) {
            int bin = static_cast<int>((v - d_min) / bin_width);
            if (bin >= n_bins) bin = n_bins - 1;
            if (bin < 0) bin = 0;
            counts[bin]++;
        }

        int max_count = *std::max_element(counts.begin(), counts.end());

        BrailleCanvas canvas(width, height);
        int pw = canvas.pixel_width();
        int ph = canvas.pixel_height();

        // Draw histogram bars
        double bar_width = static_cast<double>(pw) / n_bins;

        for (int b = 0; b < n_bins; ++b) {
            int bar_height = static_cast<int>(static_cast<double>(counts[b]) / max_count * ph);
            int x_start = static_cast<int>(b * bar_width);
            int x_end = static_cast<int>((b + 1) * bar_width) - 1;

            for (int y = ph - bar_height; y < ph; ++y) {
                for (int x = x_start; x <= x_end; ++x) {
                    canvas.set_pixel(x, y);
                }
            }
        }

        std::stringstream ss;
        ss << params.value("title", "Histogram") << "\n";
        ss << "count: [0, " << max_count << "]\n";
        ss << canvas.render();
        ss << "value: [" << d_min << ", " << d_max << "]";

        json out;
        out["plot"] = ss.str();
        out["format"] = "braille";
        out["n_bins"] = n_bins;
        out["max_count"] = max_count;
        out["data_range"] = {d_min, d_max};

        result.success = true;
        result.data = out.dump();
        return result;
    }

    // Plot mathematical function
    Result plot_function(const json& params, int width, int height) {
        Result result;

        std::string func = params.value("function", "sin");
        double x_min = params.value("x_min", -3.14159);
        double x_max = params.value("x_max", 3.14159);
        int n_points = params.value("n_points", 200);

        // Generate x values
        std::vector<double> x(n_points), y(n_points);
        double dx = (x_max - x_min) / (n_points - 1);

        for (int i = 0; i < n_points; ++i) {
            x[i] = x_min + i * dx;

            // Evaluate function
            if (func == "sin") {
                y[i] = std::sin(x[i]);
            } else if (func == "cos") {
                y[i] = std::cos(x[i]);
            } else if (func == "exp") {
                y[i] = std::exp(x[i]);
            } else if (func == "log") {
                y[i] = x[i] > 0 ? std::log(x[i]) : std::nan("");
            } else if (func == "x^2" || func == "square") {
                y[i] = x[i] * x[i];
            } else if (func == "x^3" || func == "cube") {
                y[i] = x[i] * x[i] * x[i];
            } else if (func == "sqrt") {
                y[i] = x[i] >= 0 ? std::sqrt(x[i]) : std::nan("");
            } else if (func == "tanh") {
                y[i] = std::tanh(x[i]);
            } else if (func == "sigmoid") {
                y[i] = 1.0 / (1.0 + std::exp(-x[i]));
            } else if (func == "gaussian") {
                y[i] = std::exp(-x[i] * x[i] / 2);
            } else {
                y[i] = std::sin(x[i]);  // Default
            }
        }

        // Remove NaN values
        std::vector<double> x_clean, y_clean;
        for (int i = 0; i < n_points; ++i) {
            if (!std::isnan(y[i]) && !std::isinf(y[i])) {
                x_clean.push_back(x[i]);
                y_clean.push_back(y[i]);
            }
        }

        // Create params for line plot
        json line_params;
        line_params["x"] = x_clean;
        line_params["y"] = y_clean;
        line_params["title"] = "f(x) = " + func;

        return plot_xy(line_params, width, height, true);
    }
};

std::unique_ptr<Command> create_plot_braille() {
    return std::make_unique<PlotBrailleCommand>();
}

} // namespace math_tools
