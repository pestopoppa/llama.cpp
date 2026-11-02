#include "../tools/server/server-native-correlation.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <atomic>
#include <cassert>
#include <thread>
#include <vector>

int main() {
    const std::string id = "0123456789abcdef0123456789abcdef";
    std::string value;
    assert(server_parse_native_correlation({}, value) == server_correlation_parse::absent);
    assert(server_parse_native_correlation({{"X-Llama-Native-Correlation-Id", id}}, value) == server_correlation_parse::valid);
    assert(value == id);
    assert(server_parse_native_correlation({{"x-llama-native-correlation-id", "bad"}}, value) == server_correlation_parse::invalid);
    assert(server_parse_native_correlation({{"x-llama-native-correlation-id", id},
            {"X-Llama-Native-Correlation-Id", id}}, value) == server_correlation_parse::invalid);
    assert(!server_erase_has_native_correlation({}, {{"action", "erase"}}));
    assert(server_erase_has_native_correlation({{"X-Llama-Native-Correlation-Id", id}}, {}));
    assert(server_erase_has_native_correlation({{"X-Llama-Native-Correlation-Id", "bad"}}, {}));
    assert(server_erase_has_native_correlation({}, {{"native_correlation_id", id}}));
    assert(server_erase_has_native_correlation({}, {{"expected_native_correlation_id", id}}));

    server_native_correlation_registry registry(2);
    std::atomic<int> accepted{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 16; ++i) {
        threads.emplace_back([&] { if (registry.reserve(id)) ++accepted; });
    }
    for (auto & thread : threads) thread.join();
    assert(accepted == 1);
    assert(!registry.reserve(id));
    assert(!registry.reserve("bad"));
    assert(registry.reserve("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
    assert(!registry.reserve("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"));
    server_native_correlation_registry restarted(2);
    assert(restarted.reserve(id)); // a new incarnation has a different instance ID
}
