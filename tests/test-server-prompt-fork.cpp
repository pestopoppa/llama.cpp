#include "../tools/server/server-task.h"

int main() {
    server_prompt src;
    server_prompt dst;
    if (!src.can_seed_exact_fork()) return 1;
    src.approx = true;
    src.approx_from = 32;
    if (src.can_seed_exact_fork()) return 2;
    dst.approx = true;
    dst.approx_from = 8;
    // A refused source does not alter the destination.
    if (src.can_seed_exact_fork()) dst.mark_exact_fork();
    if (!dst.approx || dst.approx_from != 8) return 3;
    src.approx = false;
    if (src.can_seed_exact_fork()) dst.mark_exact_fork();
    if (dst.approx || dst.approx_from != -1) return 4;
    return 0;
}
