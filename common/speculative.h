#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// tree topology built during tree-based speculative drafting
struct speculation_tree {
    std::vector<int32_t>    parent;     // parent[i] = -1 for root
    std::vector<llama_token> tokens;    // token at each node
    std::vector<float>      log_probs;  // cumulative log probability to reach node i
    int32_t n_nodes = 0;

    // enumerate all root-to-leaf paths as sequences of node indices
    std::vector<std::vector<int32_t>> get_paths() const;

    // get the best (highest cumulative log prob) root-to-leaf path as tokens
    llama_tokens get_best_path() const;

    // get the greedy path (follow primary/top-1 child at each depth) as tokens
    llama_tokens get_greedy_path() const;
};

// get tree from last draft call (nullptr if no tree or p_split == 0)
const speculation_tree * common_speculative_get_tree(const common_speculative * spec);

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// check if the llama_context is compatible for speculative decoding
// note: clears the memory of the context
bool common_speculative_is_compat(llama_context * ctx_tgt);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);
