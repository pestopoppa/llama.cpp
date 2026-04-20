/**
 * @file plot_sixel.cpp
 * @brief High-resolution Sixel graphics plotting
 *
 * Creates plots using the Sixel graphics protocol for terminals that support it.
 * Falls back to braille for unsupported terminals.
 *
 * Usage:
 *   {"command": "plot_sixel", "x": [0,1,2,3,4], "y": [0,1,4,9,16], "title": "y = x²"}
 */

#include "../../include/common.hpp"
#include "../../include/json_io.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <array>
#include <cstdlib>

namespace math_tools {

class PlotSixelCommand : public Command {
public:
    std::string name() const override { return "plot_sixel"; }
    std::string description() const override {
        return "Create high-resolution sixel graphics plot";
    }

    Result execute(const std::string& params_json) override {
        Timer timer;
        Result result;

        try {
            json params = json::parse(params_json);

            if (!params.contains("x") || !params.contains("y")) {
                result.error = "Missing required parameters: x and y";
                return result;
            }

            std::vector<double> x = params["x"].get<std::vector<double>>();
            std::vector<double> y = params["y"].get<std::vector<double>>();

            if (x.size() != y.size()) {
                result.error = "x and y must have the same length";
                return result;
            }
            if (x.empty()) {
                result.error = "x and y cannot be empty";
                return result;
            }

            std::string plot_type = params.value("type", "line");
            int width = params.value("width", 800);
            int height = params.value("height", 400);
            std::string title = params.value("title", "");

            width = std::clamp(width, 100, 2000);
            height = std::clamp(height, 50, 1000);

            bool sixel_supported = checkSixelSupport();

            std::string output;
            std::string format;

            if (sixel_supported) {
                output = generateSixel(x, y, plot_type, width, height, title);
                format = "sixel";
            } else {
                output = generateBraille(x, y, plot_type, width / 10, height / 5, title);
                format = "braille";
            }

            double x_min = *std::min_element(x.begin(), x.end());
            double x_max = *std::max_element(x.begin(), x.end());
            double y_min = *std::min_element(y.begin(), y.end());
            double y_max = *std::max_element(y.begin(), y.end());

            json data;
            if (format == "sixel") {
                data["sixel"] = output;
            } else {
                data["plot"] = output;
            }
            data["format"] = format;
            data["dimensions"]["width"] = width;
            data["dimensions"]["height"] = height;
            data["x_range"] = {x_min, x_max};
            data["y_range"] = {y_min, y_max};
            data["n_points"] = static_cast<int>(x.size());

            result.success = true;
            result.data = data.dump();

        } catch (const std::exception& e) {
            result.error = std::string("Plot error: ") + e.what();
        }

        result.elapsed_ms = timer.elapsed_ms();
        return result;
    }

private:
    bool checkSixelSupport() {
        const char* term = std::getenv("TERM");
        if (!term) return false;

        std::string term_str(term);
        if (term_str.find("xterm") != std::string::npos ||
            term_str.find("mlterm") != std::string::npos ||
            term_str.find("mintty") != std::string::npos ||
            term_str.find("wezterm") != std::string::npos ||
            term_str.find("foot") != std::string::npos) {
            return true;
        }
        return false;
    }

    std::string generateSixel(const std::vector<double>& x,
                               const std::vector<double>& y,
                               const std::string& plot_type,
                               int width, int height,
                               const std::string& title) {
        std::vector<std::vector<std::array<uint8_t, 3>>> pixels(
            height, std::vector<std::array<uint8_t, 3>>(width, {255, 255, 255}));

        double x_min = *std::min_element(x.begin(), x.end());
        double x_max = *std::max_element(x.begin(), x.end());
        double y_min = *std::min_element(y.begin(), y.end());
        double y_max = *std::max_element(y.begin(), y.end());

        int margin_left = 50;
        int margin_right = 20;
        int margin_top = title.empty() ? 10 : 30;
        int margin_bottom = 30;

        int plot_width = width - margin_left - margin_right;
        int plot_height = height - margin_top - margin_bottom;

        double x_scale = (x_max > x_min) ? plot_width / (x_max - x_min) : 1.0;
        double y_scale = (y_max > y_min) ? plot_height / (y_max - y_min) : 1.0;

        std::array<uint8_t, 3> axis_color = {128, 128, 128};
        int y_axis = height - margin_bottom;
        for (int px = margin_left; px < width - margin_right; ++px) {
            pixels[y_axis][px] = axis_color;
        }
        for (int py = margin_top; py < height - margin_bottom; ++py) {
            pixels[py][margin_left] = axis_color;
        }

        std::array<uint8_t, 3> data_color = {0, 100, 200};

        if (plot_type == "scatter") {
            for (size_t i = 0; i < x.size(); ++i) {
                int px = margin_left + static_cast<int>((x[i] - x_min) * x_scale);
                int py = height - margin_bottom - static_cast<int>((y[i] - y_min) * y_scale);
                px = std::clamp(px, margin_left, width - margin_right - 1);
                py = std::clamp(py, margin_top, height - margin_bottom - 1);

                for (int dy = -2; dy <= 2; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        if (dx*dx + dy*dy <= 4) {
                            int pxx = px + dx;
                            int pyy = py + dy;
                            if (pxx >= 0 && pxx < width && pyy >= 0 && pyy < height) {
                                pixels[pyy][pxx] = data_color;
                            }
                        }
                    }
                }
            }
        } else {
            for (size_t i = 0; i + 1 < x.size(); ++i) {
                int x1 = margin_left + static_cast<int>((x[i] - x_min) * x_scale);
                int y1 = height - margin_bottom - static_cast<int>((y[i] - y_min) * y_scale);
                int x2 = margin_left + static_cast<int>((x[i+1] - x_min) * x_scale);
                int y2 = height - margin_bottom - static_cast<int>((y[i+1] - y_min) * y_scale);
                drawLine(pixels, x1, y1, x2, y2, data_color);
            }
        }

        return pixelsToSixel(pixels);
    }

    void drawLine(std::vector<std::vector<std::array<uint8_t, 3>>>& pixels,
                  int x1, int y1, int x2, int y2,
                  const std::array<uint8_t, 3>& color) {
        int height = pixels.size();
        int width = pixels[0].size();

        int dx = std::abs(x2 - x1);
        int dy = std::abs(y2 - y1);
        int sx = (x1 < x2) ? 1 : -1;
        int sy = (y1 < y2) ? 1 : -1;
        int err = dx - dy;

        while (true) {
            if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
                pixels[y1][x1] = color;
                if (y1 + 1 < height) pixels[y1 + 1][x1] = color;
                if (x1 + 1 < width) pixels[y1][x1 + 1] = color;
            }
            if (x1 == x2 && y1 == y2) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x1 += sx; }
            if (e2 < dx) { err += dx; y1 += sy; }
        }
    }

    std::string pixelsToSixel(const std::vector<std::vector<std::array<uint8_t, 3>>>& pixels) {
        int height = pixels.size();
        int width = pixels[0].size();

        std::ostringstream ss;
        ss << "\033Pq";
        ss << "#0;2;100;100;100";
        ss << "#1;2;0;40;80";
        ss << "#2;2;50;50;50";

        for (int row = 0; row < height; row += 6) {
            for (int color_id = 1; color_id <= 2; ++color_id) {
                ss << "#" << color_id;
                for (int col = 0; col < width; ++col) {
                    int sixel_val = 0;
                    for (int bit = 0; bit < 6 && row + bit < height; ++bit) {
                        const auto& pixel = pixels[row + bit][col];
                        bool is_this_color = false;
                        if (color_id == 1 && pixel[2] > 150) is_this_color = true;
                        else if (color_id == 2 && pixel[0] == 128 && pixel[1] == 128) is_this_color = true;
                        if (is_this_color) sixel_val |= (1 << bit);
                    }
                    ss << static_cast<char>(sixel_val + 63);
                }
                ss << "$";
            }
            ss << "-";
        }
        ss << "\033\\";
        return ss.str();
    }

    std::string generateBraille(const std::vector<double>& x,
                                 const std::vector<double>& y,
                                 const std::string& plot_type,
                                 int width, int height,
                                 const std::string& title) {
        int dot_width = width * 2;
        int dot_height = height * 4;
        std::vector<std::vector<bool>> dots(dot_height, std::vector<bool>(dot_width, false));

        double x_min = *std::min_element(x.begin(), x.end());
        double x_max = *std::max_element(x.begin(), x.end());
        double y_min = *std::min_element(y.begin(), y.end());
        double y_max = *std::max_element(y.begin(), y.end());

        double x_scale = (x_max > x_min) ? (dot_width - 1) / (x_max - x_min) : 1.0;
        double y_scale = (y_max > y_min) ? (dot_height - 1) / (y_max - y_min) : 1.0;

        for (size_t i = 0; i < x.size(); ++i) {
            int px = static_cast<int>((x[i] - x_min) * x_scale);
            int py = dot_height - 1 - static_cast<int>((y[i] - y_min) * y_scale);
            px = std::clamp(px, 0, dot_width - 1);
            py = std::clamp(py, 0, dot_height - 1);
            dots[py][px] = true;
        }

        if (plot_type == "line") {
            for (size_t i = 0; i + 1 < x.size(); ++i) {
                int x1 = static_cast<int>((x[i] - x_min) * x_scale);
                int y1 = dot_height - 1 - static_cast<int>((y[i] - y_min) * y_scale);
                int x2 = static_cast<int>((x[i+1] - x_min) * x_scale);
                int y2 = dot_height - 1 - static_cast<int>((y[i+1] - y_min) * y_scale);

                int steps = std::max(std::abs(x2 - x1), std::abs(y2 - y1));
                if (steps > 0) {
                    for (int s = 0; s <= steps; ++s) {
                        int px = x1 + (x2 - x1) * s / steps;
                        int py = y1 + (y2 - y1) * s / steps;
                        px = std::clamp(px, 0, dot_width - 1);
                        py = std::clamp(py, 0, dot_height - 1);
                        dots[py][px] = true;
                    }
                }
            }
        }

        std::ostringstream ss;
        if (!title.empty()) ss << title << "\n";

        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                int pattern = 0;
                int base_y = row * 4;
                int base_x = col * 2;

                if (base_y < dot_height && base_x < dot_width && dots[base_y][base_x]) pattern |= 0x01;
                if (base_y + 1 < dot_height && base_x < dot_width && dots[base_y + 1][base_x]) pattern |= 0x02;
                if (base_y + 2 < dot_height && base_x < dot_width && dots[base_y + 2][base_x]) pattern |= 0x04;
                if (base_y < dot_height && base_x + 1 < dot_width && dots[base_y][base_x + 1]) pattern |= 0x08;
                if (base_y + 1 < dot_height && base_x + 1 < dot_width && dots[base_y + 1][base_x + 1]) pattern |= 0x10;
                if (base_y + 2 < dot_height && base_x + 1 < dot_width && dots[base_y + 2][base_x + 1]) pattern |= 0x20;
                if (base_y + 3 < dot_height && base_x < dot_width && dots[base_y + 3][base_x]) pattern |= 0x40;
                if (base_y + 3 < dot_height && base_x + 1 < dot_width && dots[base_y + 3][base_x + 1]) pattern |= 0x80;

                char32_t braille = 0x2800 + pattern;
                ss << static_cast<char>(0xE0 | (braille >> 12));
                ss << static_cast<char>(0x80 | ((braille >> 6) & 0x3F));
                ss << static_cast<char>(0x80 | (braille & 0x3F));
            }
            ss << "\n";
        }
        return ss.str();
    }
};

std::unique_ptr<Command> create_plot_sixel() {
    return std::make_unique<PlotSixelCommand>();
}

} // namespace math_tools
