#include "../tools/server/server-slot-erase.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <climits>

int main() {
    int id = -1;
    assert(server_parse_expected_task_id("0", id) && id == 0);
    assert(server_parse_expected_task_id("2147483647", id) && id == INT_MAX);
    for (const auto & invalid : {"", "-0", "-1", "+1", " 1", "1 ", "1x", "2147483648"}) {
        id = 27;
        assert(!server_parse_expected_task_id(invalid, id));
        assert(id == 27);
    }

    assert(server_parse_conditional_erase(false, "", false, "", id) && id == -1);
    assert(!server_parse_conditional_erase(true, "7", false, "", id));
    assert(!server_parse_conditional_erase(false, "", true, "instance", id));
    assert(!server_parse_conditional_erase(true, "7", true, "", id));
    assert(server_parse_conditional_erase(true, "7", true, "instance", id) && id == 7);

    // A slot may be reused between a client observation and task-loop handling.
    assert(!server_slot_erase_matches(7, "old", "new", true, true, 7));
    assert(!server_slot_erase_matches(7, "", "new", true, true, 7));
    assert(!server_slot_erase_matches(7, "new", "new", true, true, 8));
    assert(!server_slot_erase_matches(7, "new", "new", false, false, -1));
    assert(!server_slot_erase_matches(7, "new", "new", true, false, -1));
    assert(server_slot_erase_matches(7, "new", "new", true, true, 7));
    assert(server_slot_erase_matches(-1, "", "new", false, false, -1));
    assert(server_slot_erase_matches(-1, "", "new", true, true, 8));
}
