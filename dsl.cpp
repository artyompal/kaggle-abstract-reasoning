
#include <array>
#include <algorithm>
#include <atomic>
#include <bitset>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

const int NUM_COLORS = 10;
const int MAX_DIMS = 30;

#include "json11.hpp"
#include "helpers.h"

#if !ENABLE_MAIN
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
#endif // ENABLE_MAIN

#include "static_array.h"
#include "algorithms.h"
#include "dsl.h"
#include "json11.cpp"


struct Point {
    uint x, y;

    Point() : x{}, y{} {}
    Point(uint a, uint b) : x{a}, y{b} {}

    bool operator <(const Point &another) const {
        return (x != another.x) ? x < another.x : y < another.y;
    }
};

using Island = std::vector<Point>;
const int MAX_AREA = MAX_DIMS * MAX_DIMS;


enum RuleType {
    COPY_COLOR_BY_DIRECTION,
    CORNER_CHECK,
    NBH_CHECK,
    DIRECT_CHECK,
    INDIRECT_CHECK,
    COLOR_DISTRIBUTION,

    FLIP,
    ROTATE,
    DISTRIBUTE_FROM_BORDER,
    COLOR_FOR_INNERS,
    DRAW_LINES,
    DRAW_LINE_TO,
    DISTRIBUTE_COLORS,
    UNITY,
    SPLIT_BY_H,
    SPLIT_BY_W,
    MAP_COLOR,
    CROP_EMPTY,
    CROP_FIGURE,
    MAKE_HOLES,
    GRAVITY,

    CELLS,
    FIGURES,

    NOTHING,
    COLOR_FIGURES,

    CELLWISE_OR,
    OUTPUT_FIRST,
    OUTPUT_LAST,

    ALIGN_PATTERN,

    REDUCE,
    MACRO_MULTIPLY,
};

enum Direction {
    // Next 8 values must occupy places from 0 to 7
    TOP,
    BOTTOM,
    LEFT,
    RIGHT,

    TOP_LEFT,
    BOTTOM_LEFT,
    TOP_RIGHT,
    BOTTOM_RIGHT,

    EVERYWHERE,

    HORIZONTAL,
    VERTICAL,
    HORVER,
    DIAGONAL,

    NONE,
    BORDER,
    COLOR,

    BIGGEST,
    SMALLEST,

    ALL,
    INDEX,
    LAST,

    VER,
    HOR,
};

enum MergeRule {
    OR,
    AND,
    EQUAL,
    XOR,
};

enum MacroType {
    GLOBAL_RULE,
    GLOBAL_INTERACTION_RULE,
    CA_RULE,
};

#include "dsl_tables.h"

struct Rule {
    MacroType macro_type;

    std::bitset<NUM_COLORS> ignore_colors;
    RuleType type;
    Direction direction;

    std::bitset<NUM_COLORS> copy_color;
    data_type look_back_color;

    std::bitset<NUM_COLORS> nbh_check_colors;
    data_type nbh_check_out;
    uint nbh_check_sum;

    int check_in_empty;
    data_type color_in;
    data_type color_out;

    std::bitset<NUM_COLORS> colors;

    data_type start_by_color;
    data_type not_stop_by_color;
    data_type with_color;

    Direction mode;

    int horizontally;
    int vertically;
    data_type intersect;

    RuleType gravity_type;
    bool look_at_what_to_move;
    data_type color_what;
    Direction direction_type;
    Direction direction_border;
    data_type direction_color;
    int steps_limit;

    MergeRule merge_rule;

    Direction sort;
    data_type allow_color;
    Direction apply_to;
    data_type apply_to_index;

    Direction how;
    data_type rotations_count;

    data_type dif_c_edge;

    data_type k;
    data_type k1;
    data_type k2;
    data_type skip_color;
    data_type not_stop_by_color_and_skip;
    data_type fill_with_color;
};

using Rules = std::vector<Rule>;

struct AutomatonParams {
    Rules global_rules;
    Rules ca_rules;
    Rule split_rule;
    Rule merge_rule;
};


enum NeisMode {
    NeisAll,
    NeisDirect,
    NeisIndirect,

    NeisUpLeft,
    NeisUpRight,
    NeisBottomLeft,
    NeisBottomRight,
};

using NeisArray = static_array<data_type, 8>;   // there could be 8 neighbours at most


std::vector<Island> get_connectivity_info(const AutomatonState &input, bool ignore_black,
                                          bool edge_for_difcolors = false);

AutomatonState apply_rule(const AutomatonState &input, const Rule &rule);
AutomatonState compute_parametrized_automata(const AutomatonState &input, const Rules &rules);


const std::array<std::array<int, 2>, 8> all_neis = {{
    {-1, 0}, {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}}};
const std::array<std::array<int, 2>, 4> direct_neis = {{
    {-1, 0}, {0, -1}, {1, 0}, {0, 1}}};
const std::array<std::array<int, 2>, 4> indirect_neis = {{
    {-1, -1}, {1, -1}, {1, 1}, {-1, 1}}};

const std::array<std::array<int, 2>, 3> up_left_neis = {{
    {-1, -1}, {-1, 0}, {0, -1}}};
const std::array<std::array<int, 2>, 3> up_right_neis = {{
    {-1, 1}, {-1, 0}, {0, 1}}};
const std::array<std::array<int, 2>, 3> btm_left_neis = {{
    {1, -1}, {1, 0}, {0, -1}}};
const std::array<std::array<int, 2>, 3> btm_right_neis = {{
    {1, 1}, {1, 0}, {0, 1}}};


NeisArray get_neighbours(const AutomatonState &state, int i, int j, NeisMode mode) {
    NeisArray res;
    const std::array<int, 2> *neis = nullptr;
    uint neis_count = 0;

    switch (mode) {
    case NeisAll:
        neis = all_neis.data(), neis_count = all_neis.size();
        break;
    case NeisDirect:
        neis = direct_neis.data(), neis_count = direct_neis.size();
        break;
    case NeisIndirect:
        neis = indirect_neis.data(), neis_count = indirect_neis.size();
        break;

    case NeisUpLeft:
        neis = up_left_neis.data(), neis_count = up_left_neis.size();
        break;
    case NeisUpRight:
        neis = up_right_neis.data(), neis_count = up_right_neis.size();
        break;
    case NeisBottomLeft:
        neis = btm_left_neis.data(), neis_count = btm_left_neis.size();
        break;
    case NeisBottomRight:
        neis = btm_right_neis.data(), neis_count = btm_right_neis.size();
        break;

    default:
        assert(false);
    }

    for (uint k = 0; k < neis_count; k++) {
        uint a = i + neis[k][0], b = j + neis[k][1];

        if (a < state.shape[0] && b < state.shape[1]) {
            res.push_back(state(a, b));
        }
    }

    return res;
}


// def apply_split_rule(input, hidden, split_rule):
std::vector<AutomatonState> apply_split_rule(const AutomatonState &input, const Rule &split_rule) {
    // if split_rule['type'] == 'nothing':
    //     return [(input, hidden)]
    if (split_rule.type == NOTHING) {
        return {input};
    }

    // if split_rule['type'] == 'macro_multiply':
    //     ks = split_rule['k1'] *  split_rule['k2']
    //     grids = [(np.copy(input), np.copy(hidden)) for _ in range(ks)]
    //     return grids
    if (split_rule.type == MACRO_MULTIPLY) {
        std::vector<AutomatonState> res;
        uint ks = split_rule.k1 * split_rule.k2;

        res.reserve(ks);

        for (uint i = 0; i < ks; i++) {
            res.push_back(input);
        }

        return res;
    }

    // split_rule['type'] = 'figures'
    // dif_c_edge = split_rule['type'] == 'figures'
    // communities = get_connectivity_info(input, ignore_black=True, edge_for_difcolors=dif_c_edge)
    // communities = sorted(communities, key = len)
    // if split_rule['sort'] == 'biggest':
    //     communities = communities[::-1]
    bool dif_c_edge = split_rule.type == FIGURES;
    std::vector<Island> communities = get_connectivity_info(input, true, dif_c_edge);

    if (split_rule.sort == BIGGEST) {
        std::reverse(communities.begin(), communities.end());
    }

    // grids = [(np.zeros_like(input), np.zeros_like(hidden)) for _ in range(len(communities))]
    std::vector<AutomatonState> grids;
    grids.reserve(communities.size());

    // for i in range(len(communities)):
    //     for point in communities[i]:
    //         grids[i][0][point] = input[point]
    for (uint i = 0; i < communities.size(); i++) {
        grids.emplace_back(input.shape);
        AutomatonState &last = grids.back();

        for (const Point &p : communities[i]) {
            last(p.x, p.y) = input(p.x, p.y);
        }
    }

    if (grids.empty()) {
        grids.push_back(input);
    }

    return grids;
}

// def apply_merge_rule(grids, merge_rule):
AutomatonState apply_merge_rule(std::vector<AutomatonState> grids, const Rule &merge_rule,
                                const Rule &split_rule) {
    // if split_rule['type'] == 'macro_multiply':
    //     shape_base = grids[0][0].shape
    //     shapes = [arr[0].shape for arr in grids]
    //     if not np.array([shape_base == sh for sh in shapes]).all():
    //         return np.zeros((1))
    //
    //     ks_1 = split_rule['k1']
    //     ks_2 = split_rule['k2']
    //     output = np.zeros((shape_base[0] * ks_1, shape_base[1] * ks_2))
    //     for k1 in range(ks_1):
    //         for k2 in range(ks_2):
    //             output[(k1*shape_base[0]):((k1+1) * shape_base[0]),
    //                    (k2*shape_base[1]):((k2+1) * shape_base[1])] = grids[k1*ks_2 + k2][0]
    //
    //     return output
    if (split_rule.type == MACRO_MULTIPLY) {
        const auto &shape_base = grids[0].shape;
        for (uint i = 1; i < grids.size(); i++) {
            if (grids[i].shape != shape_base) {
                return AutomatonState{1, 1};
            }
        }

        uint ks_1 = split_rule.k1, ks_2 = split_rule.k2;
        AutomatonState output{shape_base[0] * ks_1, shape_base[1] * ks_2};

        for (uint k1 = 0; k1 < ks_1; k1++) {
            for (uint k2 = 0; k2 < ks_2; k2++) {
                const AutomatonState &grid = grids[(k1 * ks_2 + k2) % grids.size()];

                for (uint i = 0; i < shape_base[0]; i++) {
                    for (uint j = 0; j < shape_base[1]; j++) {
                        output(k1 * shape_base[0] + i, k2 * shape_base[1] + j) = grid(i, j);
                    }
                }
            }
        }

        return output;
    }

    // if merge_rule['type'] == 'cellwise_or':
    //     output = np.zeros_like(grids[0][0])
    //     for i in np.arange(len(grids))[::-1]:
    //         if grids[i][0].shape == output.shape:
    //             output[grids[i][0]>0] = grids[i][0][grids[i][0]>0]
    //     return output
    // elif merge_rule['type'] == 'output_first':
    //     output = grids[0][0]
    // elif merge_rule['type'] == 'output_last':
    //     output = grids[-1][0]
    // return output

    if (merge_rule.type == CELLWISE_OR) {
        AutomatonState output{grids[0].shape};

        for (uint grid_idx = grids.size() - 1; grid_idx != UINT_MAX; grid_idx--) {
            if (grids[grid_idx].shape == output.shape) {
                for (uint i = 0; i < grids[grid_idx].shape[0]; i++) {
                    for (uint j = 0; j < grids[grid_idx].shape[1]; j++) {
                        data_type col = grids[grid_idx](i, j);

                        if (col > 0) {
                            output(i, j) = col;
                        }
                    }
                }
            }
        }

        return output;
    } else if (merge_rule.type == OUTPUT_FIRST) {
        return grids.front();
    } else if (merge_rule.type == OUTPUT_LAST) {
        return grids.back();
    } else {
        assert(false);
        return grids.front();
    }
}

// def apply_interaction_rule(grids, rule):
void apply_interaction_rule(std::vector<AutomatonState> &grids, const Rule &rule) {
    // if rule['type'] == 'align_pattern':
    assert(rule.type == ALIGN_PATTERN);

    // allow_rotation = rule['allow_rotation']
    // if len(grids) > 5:
    //     return grids
    const int MAX_GRIDS_FOR_APPLY_RULE = 5;

    if (grids.size() > MAX_GRIDS_FOR_APPLY_RULE) {
        return;
    }

    // for index_from in range(len(grids)):
    //     for index_to in range(index_from+1, len(grids)):
    for (uint index_from = 0; index_from < grids.size(); index_from++) {
        for (uint index_to = index_from + 1; index_to < grids.size(); index_to++) {
            // input_i = grids[index_from][0]
            // input_j = grids[index_to][0]
            AutomatonState &input_i = grids[index_from];
            AutomatonState &input_j = grids[index_to];

            // i_nonzero_rows = np.arange(input_i.shape[0])[np.max(input_i>0, axis=1)]
            // i_nonzero_columns = np.arange(input_i.shape[1])[np.max(input_i>0, axis=0)]
            // j_nonzero_rows = np.arange(input_j.shape[0])[np.max(input_j>0, axis=1)]
            // j_nonzero_columns = np.arange(input_j.shape[1])[np.max(input_j>0, axis=0)]
            //
            // if i_nonzero_rows.shape[0] == 0 or i_nonzero_columns.shape[0] == 0 or
            //                 j_nonzero_rows.shape[0] == 0 or j_nonzero_columns.shape[0] == 0:
            //     continue
            //
            // i_minrow = np.min(i_nonzero_rows)
            // i_mincol = np.min(i_nonzero_columns)
            // i_maxrow = np.max(i_nonzero_rows) + 1
            // i_maxcol = np.max(i_nonzero_columns) + 1
            // j_minrow = np.min(j_nonzero_rows)
            // j_mincol = np.min(j_nonzero_columns)
            // j_maxrow = np.max(j_nonzero_rows) + 1
            // j_maxcol = np.max(j_nonzero_columns) + 1


            // This is not optimal:
            //
            // static_array<uint, MAX_DIMS> i_nonzero_rows = get_nonzero_rows(input_i);
            // static_array<uint, MAX_DIMS> i_nonzero_columns = get_nonzero_cols(input_i);
            // static_array<uint, MAX_DIMS> j_nonzero_rows = get_nonzero_rows(input_j);
            // static_array<uint, MAX_DIMS> j_nonzero_columns = get_nonzero_cols(input_j);
            //
            // if (i_nonzero_rows.empty() or i_nonzero_columns.empty() or
            //     j_nonzero_rows.empty() or j_nonzero_columns.empty()) {
            //         continue;
            //     }

            uint i_minrow, i_mincol, i_maxrow, i_maxcol;
            get_bounding_box(i_minrow, i_mincol, i_maxrow, i_maxcol, input_i);
            if (i_minrow >= i_maxrow or i_mincol >= i_maxcol) {
                continue;
            }

            uint j_minrow, j_mincol, j_maxrow, j_maxcol;
            get_bounding_box(j_minrow, j_mincol, j_maxrow, j_maxcol, input_j);
            if (j_minrow >= j_maxrow or j_mincol >= j_maxcol) {
                continue;
            }

            // figure_to_align = input_i[i_minrow:i_maxrow, i_mincol:i_maxcol]
            // figure_target = input_j[j_minrow:j_maxrow, j_mincol:j_maxcol]
            std::array<uint, 2> figure_to_align_shape = {i_maxrow - i_minrow, i_maxcol - i_mincol};
            std::array<uint, 2> figure_target_shape = {j_maxrow - j_minrow, j_maxcol - j_mincol};

            // best_fit = 0
            // best_i_fit, best_j_fit = -1, -1
            uint best_fit = 0, best_i_fit = 0, best_j_fit = 0;


            // if figure_to_align.shape[0] < figure_target.shape[0] or figure_to_align.shape[1] < figure_target.shape[1]:
            //     continue
            if (figure_to_align_shape[0] < figure_target_shape[0] or figure_to_align_shape[1] < figure_target_shape[1]) {
                continue;
            // else:
            } else {
                // for i_start in range((figure_to_align.shape[0] - figure_target.shape[0])+1):
                //     for j_start in range((figure_to_align.shape[1] - figure_target.shape[1])+1):
                //         fig_1 = figure_to_align[i_start:(i_start + figure_target.shape[0]), j_start:(j_start + figure_target.shape[1])]
                //         if np.logical_and(
                //              np.logical_and(figure_target > 0, figure_target!=rule['allow_color']),
                //              figure_target != fig_1).any():
                //             continue
                //         fit = np.sum(figure_target==fig_1)
                //         if fit > best_fit:
                //             best_i_fit, best_j_fit = i_start, j_start
                //             best_fit = fit
                for (uint i_start = 0; i_start < figure_to_align_shape[0] - figure_target_shape[0] + 1; i_start++) {
                    for (uint j_start = 0; j_start < figure_to_align_shape[1] - figure_target_shape[1] + 1; j_start++) {
                        uint fit = 0;

                        for (uint a = 0; a < figure_target_shape[0]; a++) {
                            for (uint b = 0; b < figure_target_shape[1]; b++) {
                                data_type target_col = input_j(j_minrow + a, j_mincol + b);
                                data_type source_col = input_i(i_minrow + i_start + a,
                                                               i_mincol + j_start + b);

                                if (target_col > 0 and target_col != rule.allow_color and target_col != source_col) {
                                    goto reject;
                                }

                                fit += source_col == target_col;
                            }
                        }

                        if (fit > best_fit) {
                            best_fit = fit, best_i_fit = i_start, best_j_fit = j_start;
                        }

                        reject: ;
                    }
                }

                // if best_fit == 0:
                //     continue
                if (best_fit == 0) {
                    continue;
                }

                // imin = j_minrow - best_i_fit
                // imax = j_minrow - best_i_fit + figure_to_align.shape[0]
                // jmin = j_mincol - best_j_fit
                // jmax = j_mincol - best_j_fit + figure_to_align.shape[1]
                int imin = j_minrow - best_i_fit;
                int imax = j_minrow - best_i_fit + figure_to_align_shape[0];
                int jmin = j_mincol - best_j_fit;
                int jmax = j_mincol - best_j_fit + figure_to_align_shape[1];

                // begin_i = max(imin, 0)
                // begin_j = max(jmin, 0)
                // end_i = min(imax, input_j.shape[0])
                // end_j = min(jmax, input_j.shape[1])
                int begin_i = std::max(imin, 0);
                int begin_j = std::max(jmin, 0);
                uint end_i = std::min(uint(imax), input_j.shape[0]);
                uint end_j = std::min(uint(jmax), input_j.shape[1]);

                // i_fig_begin = (begin_i-imin)
                // i_fig_end = figure_to_align.shape[0]-(imax-end_i)
                // j_fig_begin = (begin_j-jmin)
                // j_fig_end = figure_to_align.shape[1]-(jmax-end_j)
                uint i_fig_begin = (begin_i-imin);
                uint i_fig_end = figure_to_align_shape[0]-(imax-end_i);
                uint j_fig_begin = (begin_j-jmin);
                uint j_fig_end = figure_to_align_shape[1]-(jmax-end_j);

                // if rule['fill_with_color'] == 0:
                //     input_j[begin_i:end_i, begin_j:end_j] = figure_to_align[i_fig_begin:i_fig_end, j_fig_begin:j_fig_end]
                // else:
                //     for i, j in product(range(end_i-begin_i + 1), range(end_j-begin_j + 1)):
                //         if input_j[begin_i + i, begin_j + j] == 0:
                //            input_j[begin_i + i, begin_j + j] = rule['fill_with_color'] * (figure_to_align[i_fig_begin + i, j_fig_begin + j])
                if (rule.fill_with_color == 0) {
                    for (uint a = i_fig_begin; a < i_fig_end; a++) {
                        for (uint b = j_fig_begin; b < j_fig_end; b++) {
                            input_j(begin_i - i_fig_begin + a, begin_j - j_fig_begin + b) = \
                                input_i(i_minrow + a, i_mincol + b);
                        }
                    }
                } else {
                    for (uint a = i_fig_begin; a < i_fig_end; a++) {
                        for (uint b = j_fig_begin; b < j_fig_end; b++) {
                            data_type &dst = input_j(begin_i - i_fig_begin + a,
                                                     begin_j - j_fig_begin + b);

                            if (dst == 0) {
                                dst = rule.fill_with_color * input_i(i_minrow + a, i_mincol + b);
                            }
                        }
                    }
                }
            }
        }
    }
}

AutomatonState trace_param_automata(const AutomatonState &input, const AutomatonParams &params,
    int n_iter = 25) {
    // Execute an automata and return all the intermediate states.
    //
    // arguments:
    //     input: initial automaton state
    //     params: automaton rules: global and CA
    // returns:
    //     final automaton state
    //
    std::vector<AutomatonState> grids = apply_split_rule(input, params.split_rule);

    for (const Rule &rule : params.global_rules) {
        for (uint i = 0; i < grids.size(); i++) {
            if (rule.macro_type == GLOBAL_RULE and (rule.apply_to == ALL or
                rule.apply_to == INDEX and i == rule.apply_to_index % grids.size() or
                rule.apply_to == LAST and i == grids.size() - 1)) {
                grids[i] = apply_rule(grids[i], rule);
            } else if (rule.macro_type == GLOBAL_INTERACTION_RULE) {
                apply_interaction_rule(grids, rule);
            }
        }
    }

    for (uint i = 0; i < grids.size(); i++) {
        AutomatonState &input = grids[i];

        for (int it = 0; it < n_iter; it++) {
            AutomatonState output = compute_parametrized_automata(input, params.ca_rules);

            if (input == output) {
                break;
            }

            input = output;
        }
    }

    return apply_merge_rule(grids, params.merge_rule, params.split_rule);
}

// def compute_parametrized_automata(input, hidden_i, rules):
AutomatonState compute_parametrized_automata(const AutomatonState &input, const Rules &rules) {
    // output = np.zeros_like(input, dtype=int)
    AutomatonState output{input.shape};

    // for i, j in product(range(input.shape[0]), range(input.shape[1])):
    for (uint i = 0; i < input.shape[0]; i++) {
        for (uint j = 0; j < input.shape[1]; j++) {
            // i_c = input[i, j]
            int i_c = input(i, j);

            // cells which are adjacent to the current one

            // i_nbh = get_neighbours(input, i, j)
            // i_direct_nbh = {k: v for k, v in i_nbh.items() if k in {(1, 0), (-1, 0), (0, 1), (0, -1)}}
            // i_indirect_nbh = {k: v for k, v in i_nbh.items() if k in {(1, 1), (-1, -1), (-1, 1), (1, -1)}}

            NeisArray i_nbh = get_neighbours(input, i, j, NeisAll);
            NeisArray i_direct_nbh = get_neighbours(input, i, j, NeisDirect);
            NeisArray i_indirect_nbh = get_neighbours(input, i, j, NeisIndirect);

            // is_top_b, is_bottom_b = i == 0, i == input.shape[0] - 1
            // is_left_b, is_right_b = j == 0, j == input.shape[1] - 1
            // is_b = is_top_b or is_bottom_b or is_left_b or is_right_b
            bool is_top_b = i == 0, is_bottom_b = i == input.shape[0] - 1;
            bool is_left_b = j == 0, is_right_b = j == input.shape[1] - 1;
            // bool is_b = is_top_b or is_bottom_b or is_left_b or is_right_b;

            // if i_c > 0:
            if (i_c > 0) {
                // output[i, j] = i_c
                output(i, j) = i_c;
            }

            // for rule in rules:
            for (const Rule &rule : rules) {
                // if i_c in rule["ignore_color"]:
                if (rule.ignore_colors.test(i_c)) {
                    continue;
                }

                switch (rule.type) {

                // if rule["type"] == "copy_color_by_direction":
                case COPY_COLOR_BY_DIRECTION: {
                    // if rule['direction'] == 'bottom' or rule['direction'] == 'everywhere':
                    //     if not is_top_b and input[i - 1, j] in rule['copy_color'] and
                    //             (i == 1 or input[i - 2, j] == rule['look_back_color']):
                    //         output[i, j] = input[i - 1, j]
                    //         break
                    if (rule.direction == BOTTOM or rule.direction == EVERYWHERE) {
                        if (!is_top_b && rule.copy_color.test(input(i - 1, j)) and
                                (i == 1 or input(i - 2, j) == rule.look_back_color)) {
                            output(i, j) = input(i - 1, j);
                            goto done;
                        }
                    }

                    // if rule["direction"] == "top" or rule["direction"] == "everywhere":
                    //     if not is_bottom_b and input[i + 1, j] in rule["copy_color"] and
                    //             (i == input.shape[0] - 2 or input[i + 2, j] == rule["look_back_color"]):
                    //         output[i, j] = input[i + 1, j]
                    //         break
                    if (rule.direction == TOP or rule.direction == EVERYWHERE) {
                        if (!is_bottom_b and rule.copy_color.test(input(i + 1, j)) and
                                (i == input.shape[0] - 2 or input(i + 2, j) == rule.look_back_color)) {
                            output(i, j) = input(i + 1, j);
                            goto done;
                        }
                    }

                    // if rule["direction"] == "right" or rule["direction"] == "everywhere":
                    //     if not is_left_b and input[i, j - 1] in rule["copy_color"] and
                    //             (j == 1 or input[i, j - 2] == rule["look_back_color"]):
                    //             (j == 1 or input[i, j - 2] == rule["look_back_color"]):
                    //         output[i, j] = input[i, j - 1]
                    //         break
                    if (rule.direction == RIGHT or rule.direction == EVERYWHERE) {
                        if (!is_left_b and rule.copy_color.test(input(i, j -1)) and
                                (j == 1 or input(i, j -2) == rule.look_back_color)) {
                            output(i, j) = input(i, j - 1);
                            goto done;
                        }
                    }

                    // if rule["direction"] == "left" or rule["direction"] == "everywhere":
                    //     if not is_right_b and input[i, j + 1] in rule["copy_color"] and
                    //             (j == input.shape[1] - 2 or input[i, j + 2] == rule["look_back_color"]):
                    //         output[i, j] = input[i, j + 1]
                    //         break
                    if (rule.direction == LEFT or rule.direction == EVERYWHERE) {
                        if (!is_right_b and rule.copy_color.test(input(i, j + 1)) and
                                (j == input.shape[1] - 2 or input(i, j + 2) == rule.look_back_color)) {
                            output(i, j) = input(i, j + 1);
                            goto done;
                        }
                    }
                }
                break;


                // elif rule["type"] == "corner_check":
                case CORNER_CHECK: {
                    // color_nbh = rule["nbh_check_colors"]
                    // sum_nbh = 3
                    // out_nbh = rule["nbh_check_out"]
                    const std::bitset<NUM_COLORS> &color_nbh = rule.nbh_check_colors;
                    uint sum_nbh = 3;
                    data_type out_nbh = rule.nbh_check_out;

                    // if sum(1 for v in i_nbh.values() if v in color_nbh) < 3:
                    if (intersection_size(i_nbh, color_nbh) < 3) {
                        continue;
                    }

                    // i_uplecorner_nbh = {k: v for k, v in i_nbh.items() if k in {(-1, -1), (-1, 0), (0, -1)}}
                    // i_upricorner_nbh = {k: v for k, v in i_nbh.items() if k in {(-1, 1), (-1, 0), (0, 1)}}
                    // i_dolecorner_nbh = {k: v for k, v in i_nbh.items() if k in {(1, -1), (1, 0), (0, -1)}}
                    // i_doricorner_nbh = {k: v for k, v in i_nbh.items() if k in {(1, 1), (1, 0), (0, 1)}}
                    //
                    // did_something = False
                    // for corner_idx in [i_uplecorner_nbh, i_upricorner_nbh, i_dolecorner_nbh, i_doricorner_nbh]:
                    //     for color in color_nbh:
                    //         if sum(1 for v in corner_idx.values() if v == color) == sum_nbh:
                    //             output[i, j] = out_nbh
                    //             did_something = True
                    //             break
                    //     if did_something:
                    //         break
                    // if did_something:
                    //     break

                    // for (data_type color: color_nbh) {
                    for (data_type color = 0; color < NUM_COLORS; color++) {
                        if (color_nbh.test(color) and (
                            get_neighbours(input, i, j, NeisUpLeft).count(color) == sum_nbh or
                            get_neighbours(input, i, j, NeisUpRight).count(color) == sum_nbh or
                            get_neighbours(input, i, j, NeisBottomLeft).count(color) == sum_nbh or
                            get_neighbours(input, i, j, NeisBottomRight).count(color) == sum_nbh)) {
                            output(i, j) = out_nbh;
                            goto done;
                        }
                    }
                }
                break;


                // elif rule["type"] == "nbh_check":
                case NBH_CHECK: {
                    // color_nbh = rule["nbh_check_colors"]
                    // sum_nbh = rule["nbh_check_sum"]
                    // out_nbh = rule["nbh_check_out"]
                    //
                    // proper_nbhs = i_nbh.values()
                    // if sum(1 for v in proper_nbhs if v in color_nbh) > sum_nbh:
                    //     output[i, j] = out_nbh
                    //     break

                    const std::bitset<NUM_COLORS> &color_nbh = rule.nbh_check_colors;
                    uint sum_nbh = rule.nbh_check_sum;
                    data_type out_nbh = rule.nbh_check_out;

                    if (i_nbh.count([&](data_type v) { return color_nbh.test(v); }) > sum_nbh) {
                        output(i, j) = out_nbh;
                        goto done;
                    }
                }
                break;

                // elif rule["type"] == "direct_check":
                case DIRECT_CHECK: {
                    // color_nbh = rule["nbh_check_colors"]
                    // sum_nbh = rule["nbh_check_sum"]
                    // out_nbh = rule["nbh_check_out"]
                    //
                    // proper_nbhs = i_direct_nbh.values()
                    // if sum(1 for v in proper_nbhs if v in color_nbh) > sum_nbh:
                    //     output[i, j] = out_nbh
                    //     break

                    const std::bitset<NUM_COLORS> &color_nbh = rule.nbh_check_colors;
                    uint sum_nbh = rule.nbh_check_sum;
                    data_type out_nbh = rule.nbh_check_out;

                    if (i_direct_nbh.count([&](data_type v) { return color_nbh.test(v); }) > sum_nbh) {
                        output(i, j) = out_nbh;
                        goto done;
                    }
                }
                break;

                // elif rule["type"] == "indirect_check":
                case INDIRECT_CHECK: {
                    // color_nbh = rule["nbh_check_colors"]
                    // sum_nbh = rule["nbh_check_sum"]
                    // out_nbh = rule["nbh_check_out"]
                    //
                    // proper_nbhs = i_indirect_nbh.values()
                    // if sum(1 for v in proper_nbhs if v in color_nbh) > sum_nbh:
                    //     output[i, j] = out_nbh
                    //     break

                    const std::bitset<NUM_COLORS> &color_nbh = rule.nbh_check_colors;
                    uint sum_nbh = rule.nbh_check_sum;
                    data_type out_nbh = rule.nbh_check_out;

                    if (i_indirect_nbh.count([&](data_type v) { return color_nbh.test(v); }) > sum_nbh) {
                        output(i, j) = out_nbh;
                        goto done;
                    }
                }
                break;


                // elif rule["type"] == "color_distribution":
                case COLOR_DISTRIBUTION: {
                    // directions = ["top", "bottom", "left", "right", "topleft", "bottomleft", "topright", "bottomright"]
                    // not_border_conditions =
                    // [
                    //     not is_top_b, not is_bottom_b, not is_left_b, not is_right_b,
                    //     not is_top_b and not is_left_b,
                    //     not is_bottom_b and not is_left_b,
                    //     not is_top_b and not is_right_b,
                    //     not is_bottom_b and not is_right_b
                    // ]
                    // index_from =
                    // [   (i-1, j), (i+1, j), (i, j-1), (i, j+1),
                    //     (i-1, j-1), (i+1, j-1), (i-1, j+1), (i+1, j+1) ]
                    //
                    // for i_dir, direction in enumerate(directions):
                    //     if rule["direction"] == direction:
                    //         if not_border_conditions[i_dir]:
                    //             if (rule["check_in_empty"] == 1 and input[index_from[i_dir]] > 0) or
                    //             (rule["check_in_empty"] == 0 and input[index_from[i_dir]] == rule["color_in"]):
                    //                 output[i, j] = rule["color_out"]

                    std::array<bool, 8> border_conditions = {
                        not is_top_b, not is_bottom_b, not is_left_b, not is_right_b,
                        not is_top_b and not is_left_b,
                        not is_bottom_b and not is_left_b,
                        not is_top_b and not is_right_b,
                        not is_bottom_b and not is_right_b
                    };
                    static const std::array<std::array<int, 2>, 8> index_from = {{
                        {-1, 0}, {+1, 0}, {0, -1}, {0, +1},
                        {-1, -1}, {+1, -1}, {-1, +1}, {+1, +1}}};

                    assert(rule.direction < border_conditions.size());

                    if (border_conditions[rule.direction]) {
                        const std::array<int, 2> &offsets = index_from[rule.direction];
                        data_type col = input(offsets[0] + i, offsets[1] + j);


                        if (rule.check_in_empty and col > 0 or
                            !rule.check_in_empty and col == rule.color_in) {
                            output(i, j) = rule.color_out;
                            goto done;
                        }
                    }
                }
                break;

                default:
                    assert(false);
                }
            }

            done:;
        }
    }

    return output;
}


struct UnionFind {
    static_array<uint, MAX_AREA> area, parent;

    UnionFind(uint sz) : area(sz, 1), parent(sz) {
        for (uint i = 0; i < sz; i++) {
            parent[i] = i;
        }
    }

    uint find(uint x) {
        uint root = x;

        while (parent[root] != root) {
            root = parent[root];
        }

        while (parent[x] != root) {
            uint next_parent = parent[x];
            parent[x] = root;
            x = next_parent;
        }

        return parent[x];
    }

    void unite(uint u, uint v) {
        int root_u = find(u), root_v = find(v);

        if (root_u != root_v) {
            int area_u = area[u], area_v = area[v];

            if (area_u < area_v) {
                std::swap(root_u, root_v);
            }

            parent[root_v] = root_u;
            area[root_u] = area_u + area_v;
        }
    }
};


std::vector<Island> get_connectivity_info(const AutomatonState &input, bool ignore_black,
                                          bool edge_for_difcolors) {
    uint size = input.color.size();
    UnionFind union_find(size);

    // combine same colors
    for (uint i = 0; i < input.shape[0]; i++) {
        for (uint j = 0; j < input.shape[1]; j++) {
            data_type col = input(i, j);

            for (uint k = 0; k < all_neis.size(); k++) {
                uint u = i + all_neis[k][0], v = j + all_neis[k][1];

                // if u >= 0 and u < nrows and v >= 0 and v < ncols and
                //     (color[u, v] == color[i, j] or
                //      (edge_for_difcolors and (color[u, v]>0) == (color[i, j]>0))):
                if (u < input.shape[0] and v < input.shape[1] and (input(u, v) == col or
                    edge_for_difcolors and (input(u, v) > 0) == (col > 0))) {
                        union_find.unite(u * input.shape[1] + v, i * input.shape[1] + j);
                }
            }
        }
    }

    static_array<int, MAX_AREA> root(size);
    for (uint i = 0; i < size; i++) root[i] = union_find.find(i);

    static_array<uint, MAX_AREA> unique_islands(size);

    for (uint i = 0; i < input.shape[0]; i++) {
        for (uint j = 0; j < input.shape[1]; j++) {
            uint idx = i * input.shape[1] + j;

            if (not ignore_black or input(i, j) > 0) {
                unique_islands[idx] = union_find.parent[idx];
            } else {
                unique_islands[idx] = UINT_MAX;
            }
        }
    }

    std::sort(unique_islands.begin(), unique_islands.end());

    uint num_islands = std::unique(unique_islands.begin(), unique_islands.end()) - unique_islands.begin();
    unique_islands.resize(num_islands);

    if (unique_islands.back() == UINT_MAX) {
        unique_islands.pop_back();
        num_islands--;
    }

    static_array<uint, MAX_AREA> parent2idx(size, UINT_MAX);

    for (uint idx = 0; idx < num_islands; idx++) {
        parent2idx[unique_islands[idx]] = idx;
    }

    std::vector<Island> islands(num_islands);

    for (uint i = 0; i < input.shape[0]; i++) {
        for (uint j = 0; j < input.shape[1]; j++) {
            uint idx = i * input.shape[1] + j;
            uint parent_idx = union_find.find(idx);
            uint island_idx = parent2idx[parent_idx];

            // UINT_MAX is an ignored index for black color
            if (island_idx != UINT_MAX) {
                // printf("idx=%d parent_idx=%d island_idx=%d\n", idx, parent_idx, island_idx);
                assert(island_idx < islands.size());
                islands[island_idx].push_back(Point{i, j});
            }
        }
    }

    // input.print();
    // puts("detected islands:");
    //
    // for (const auto &island : islands) {
    //     putchar('[');
    //
    //     for (uint i = 0; i < island.size(); i++) {
    //         printf("(%d,%d) ", island[i].x, island[i].y);
    //     }
    //
    //     puts("\b]");
    // }

    std::sort(islands.begin(), islands.end(), [](const Island &a, const Island &b) {
        if (a.size() != b.size()) {
            return a.size() < b.size();
        } else {
            return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
        }});

    return islands;
}

uint count_pixels(const AutomatonState &input, uint xmin, uint xmax, uint ymin, uint ymax,
    data_type color) {
    uint count = 0;
    assert(xmax <= input.shape[0] && ymax <= input.shape[1]);

    for (uint i = xmin; i < xmax; i++) {
        for (uint j = ymin; j < ymax; j++) {
            count += input(i, j) == color;
        }
    }

    return count;
}

struct PointSet {
    std::bitset<MAX_AREA> pixels;
    uint width;

    PointSet(const Island &island, const std::array<uint, 2> &shape) : width(shape[1]) {
        for (const Point &p : island) {
            pixels.set(p.x * width + p.y);
        }
    }

    bool has(const Point &p) const {
        return pixels.test(p.x * width + p.y);
    }
};

void crop_image(AutomatonState &output, uint xmin, uint xmax, uint ymin, uint ymax) {
    uint old_width = output.shape[1];
    uint new_height = xmax - xmin, new_width = ymax - ymin;

    for (uint i = 0; i < new_height; i++) {
        for (uint j = 0; j < new_width; j++) {
            output.color[i * new_width + j] = output.color[(i + xmin) * old_width + j + ymin];
        }
    }

    output.color.resize(new_width * new_height);
    output.shape = {new_height, new_width};
}


AutomatonState apply_rule(const AutomatonState &input, const Rule &rule) {
    // output = np.zeros_like(input, dtype=int)
    // output[:, :] = input[:, :]
    AutomatonState output{input};

    switch (rule.type) {
    // // if rule['type'] == 'macro_multiply_k':
    //     // output = np.tile(output, rule['k'])
    //     output.shape = {input.shape[0] * rule.k, input.shape[1] * rule.k};
    //
    //     for (uint i = 0; i < output.shape[0]; i++) {
    //         for (uint j = 0; j < output.shape[1]; j++) {
    //             output(i % input.shape[0], j % input.shape[1]) = input(i, j);
    //         }
    //     }
    //
    //     break;

    // elif rule['type'] == 'flip':
    case FLIP:
        // if rule['how'] == 'ver':
        //     output = output[::-1, :]
        if (rule.how == VER) {
            for (uint i = 0; i < input.shape[0]; i++) {
                for (uint j = 0; j < input.shape[1]; j++) {
                    output(input.shape[0] - 1 - i, j) = input(i, j);
                }
            }
        }
        // elif rule['how'] == 'hor':
        //     output = output[:, ::-1]
        else if (rule.how == HOR) {
            for (uint i = 0; i < input.shape[0]; i++) {
                for (uint j = 0; j < input.shape[1]; j++) {
                    output(i, input.shape[1] - 1 - j) = input(i, j);
                }
            }
        } else {
            assert(false);
        }

        break;

    // elif rule['type'] == 'rotate':
    case ROTATE: {
        // output = np.rot90(output, rule['rotations_count'])
        if (rule.rotations_count % 2 == 1) {
            std::swap(output.shape[0], output.shape[1]);
        }

        if (rule.rotations_count == 1) {
            for (uint y = 0; y < input.shape[0]; y++) {
                for (uint x = 0; x < input.shape[1]; x++) {
                    uint dst_x = y;
                    uint dst_y = input.shape[1] - 1 - x;

                    output(dst_y, dst_x) = input(y, x);
                }
            }
        } else if (rule.rotations_count == 2) {
            for (uint y = 0; y < input.shape[0]; y++) {
                for (uint x = 0; x < input.shape[1]; x++) {
                    uint dst_x = input.shape[1] - 1 - x;
                    uint dst_y = input.shape[0] - 1 - y;

                    output(dst_y, dst_x) = input(y, x);
                }
            }
        } else if (rule.rotations_count == 3) {
            for (uint y = 0; y < input.shape[0]; y++) {
                for (uint x = 0; x < input.shape[1]; x++) {
                    uint dst_x = input.shape[0] - 1 - y;
                    uint dst_y = x;

                    output(dst_y, dst_x) = input(y, x);
                }
            }
        } else {
            assert(false);
        }
    } break;

    // elif rule['type'] == 'reduce':
    case REDUCE: {
        // skip_row = np.zeros(input.shape[0])
        //
        // for i in range(1, input.shape[0]):
        //     skip_row[i] = (input[i] == input[i-1]).all() or (input[i] == rule['skip_color']).all()
        //
        // if (input[0] == rule['skip_color']).all():
        //     skip_row[0] = 1
        //
        // if np.sum(skip_row==0)>0:
        //     output = input[skip_row == 0]
        std::bitset<MAX_DIMS> skip_row;

        for (uint i = 0; i < output.shape[0]; i++) {
            uint count1 = 0, count2 = 0;

            for (uint j = 0; j < output.shape[1]; j++) {
                count1 += i > 0 and output(i, j) == output(i - 1, j);
                count2 += output(i, j) == rule.skip_color;
            }

            skip_row.set(i, (count1 == output.shape[1] or count2 == output.shape[1]));
        }

        if (skip_row.count() < output.shape[0]) {
            output = output.remove_rows(skip_row);
        }


        // skip_column = np.zeros(input.shape[1])
        //
        // for i in range(1, input.shape[1]):
        //     skip_column[i] = (input[:, i] == input[:, i-1]).all() or (input[:, i] == rule['skip_color']).all()
        //
        // if (input[:, 0] == rule['skip_color']).all():
        //     skip_column[0] = 1
        //
        // if np.sum(skip_column==0)>0:
        //     output = output[:, skip_column == 0]
        std::bitset<MAX_DIMS> skip_col;

        for (uint j = 0; j < output.shape[1]; j++) {
            uint count1 = 0, count2 = 0;

            for (uint i = 0; i < output.shape[0]; i++) {
                count1 += j > 0 and output(i, j) == output(i, j - 1);
                count2 += output(i, j) == rule.skip_color;
            }

            skip_col.set(j, (count1 == output.shape[0] or count2 == output.shape[0]));
        }

        if (skip_col.count() < output.shape[1]) {
            output = output.remove_cols(skip_col);
        }
    } break;

    // if rule["type"] == "distribute_from_border":
    case DISTRIBUTE_FROM_BORDER: {
        // for i in np.arange(1, input.shape[0]-1):
        //     if output[i, 0] in rule["colors"]:
        //         if not output[i, input.shape[1]-1] in rule["colors"] or output[i, input.shape[1]-1]==output[i, 0]:
        //             output[i] = output[i, 0]
        for (uint i = 1; i < input.shape[0] - 1; i++) {
            if (rule.colors.test(output(i, 0))) {
                data_type last = output(i, input.shape[1] - 1);

                if (!rule.colors.test(last) or last == output(i, 0)) {
                    for (uint j = 0; j < input.shape[1]; j++) {
                        output(i, j) = output(i, 0);
                    }
                }
            }
        }

        // for j in np.arange(1, input.shape[1]-1):
        //     if output[0, j] in rule["colors"]:
        //         if not output[input.shape[0]-1, j] in rule["colors"] or output[input.shape[0]-1, j]==output[0, j]:
        //             output[:, j] = output[0, j]
        for (uint j = 1; j < input.shape[1] - 1; j++) {
            if (rule.colors.test(output(0, j))) {
                data_type last = output(input.shape[0] - 1, j);

                if (!rule.colors.test(last) or last == output(0, j)) {
                    for (uint i = 0; i < input.shape[0]; i++) {
                        output(i, j) = output(0, j);
                    }
                }
            }
        }
    }
    break;


    // elif rule["type"] == "color_for_inners":
    case COLOR_FOR_INNERS: {
        // hidden = np.zeros_like(input)
        // changed = 1
        AutomatonState hidden{input.shape};
        bool changed;

        // while changed == 1:
        do {
            // changed = 0
            changed = false;

            // for i, j in product(range(input.shape[0]), range(input.shape[1])):
            for (uint i = 0; i < input.shape[0]; i++) {
                for (uint j = 0; j < input.shape[1]; j++) {
                    // i_c = input[i, j]
                    data_type i_c = input(i, j);

                    // if i_c > 0 or hidden[i, j] == 1:
                    //     continue
                    if (i_c > 0 or hidden(i, j) == 1) {
                        continue;
                    }

                    // if i == 0 or i == input.shape[0] - 1 or j == 0 or j == input.shape[1] - 1:
                    //     hidden[i, j] = 1
                    //     changed = 1
                    //     continue
                    if (i == 0 or i == input.shape[0] - 1 or j == 0 or j == input.shape[1] - 1) {
                        hidden(i, j) = 1;
                        changed = true;
                        continue;
                    }

                    // cells adjacent to the current one

                    // i_nbh = get_neighbours(hidden, i, j)
                    // i_direct_nbh = {k: v for k, v in i_nbh.items() if k in {(1, 0), (-1, 0), (0, 1), (0, -1)}}
                    NeisArray i_direct_nbh = get_neighbours(hidden, i, j, NeisDirect);

                    // if sum(1 for v in i_direct_nbh.values() if v == 1) > 0:
                    //     hidden[i, j] = 1
                    //     changed = 1
                    if (i_direct_nbh.count(0) != i_direct_nbh.size()) {
                        hidden(i, j) = 1;
                        changed = true;
                    }
                }
            }
        } while (changed);

        // output[((hidden == 0).astype(np.int) * (input == 0).astype(np.int)) == 1] = rule["color_out"]
        // hidden = np.copy(hidden)

        for (uint i = 0; i < input.color.size(); i++) {
            if (hidden.color[i] == 0 and input.color[i] == 0) {
                output.color[i] = rule.color_out;
            }
        }
    }
    break;


    // elif rule["type"] == "draw_lines":
    case DRAW_LINES: {
        std::unordered_set<Direction> directions;

        switch (rule.direction) {
        // if rule["direction"] == "everywhere":
        case EVERYWHERE:
            // directions = ["top", "bottom", "left", "right", "topleft", "bottomleft", "topright", "bottomright"]
            directions.insert({TOP, BOTTOM, LEFT, RIGHT, TOP_LEFT, BOTTOM_LEFT, TOP_RIGHT, BOTTOM_RIGHT});
            break;

        // elif rule["direction"] == "horizontal":
        case HORIZONTAL:
            // directions = ["left", "right"]
            directions.insert({LEFT, RIGHT});
            break;

        // elif rule["direction"] == "vertical":
        case VERTICAL:
            // directions = ["top", "bottom"]
            directions.insert({TOP, BOTTOM});
            break;

        // elif rule["direction"] == "horver":
        case HORVER:
            // directions = ["top", "bottom", "left", "right"]
            directions.insert({TOP, BOTTOM, LEFT, RIGHT});
            break;

        // elif rule["direction"] == "diagonal":
        case DIAGONAL:
            // directions = ["topleft", "bottomleft", "topright", "bottomright"]
            directions.insert({TOP_LEFT, BOTTOM_LEFT, TOP_RIGHT, BOTTOM_RIGHT});
            break;

        // else:
        default:
            // directions = [rule["direction"]]
            directions.insert({rule.direction});
            break;
        }


        // possible_directions = ["top", "bottom", "left", "right",
        //                        "topleft", "bottomleft", "topright", "bottomright"]
        static const std::array<Direction, 8> possible_directions = {
            TOP, BOTTOM, LEFT, RIGHT, TOP_LEFT, BOTTOM_LEFT, TOP_RIGHT, BOTTOM_RIGHT};

        // index_change = [
        //     [-1, 0], [1, 0], (0, -1), (0, 1),
        //     (-1, -1), (+1, -1), (-1, +1), (+1, +1) ]
        static const std::array<std::array<int, 2>, 8> index_change = {{
            {-1, 0}, {1, 0}, {0, -1}, {0, 1},
            {-1, -1}, {+1, -1}, {-1, +1}, {+1, +1}}};

        // for i_dir, direction in enumerate(possible_directions):
        for (uint i_dir = 0; i_dir < possible_directions.size(); i_dir++) {
            Direction direction = possible_directions[i_dir];

            // if direction in directions:
            if (directions.count(direction)) {
                // idx_ch = index_change[i_dir]
                const std::array<int, 2> &idx_ch = index_change[i_dir];

                // for i in range(input.shape[0]):
                //     for j in range(input.shape[1]):
                for (uint i = 0; i < input.shape[0]; i++) {
                    for (uint j = 0; j < input.shape[1]; j++) {
                        // if input[i, j] == rule['start_by_color']:
                        //     tmp_i = i + idx_ch[0]
                        //     tmp_j = j + idx_ch[1]
                        //     while 0 <= tmp_i < input.shape[0] and
                        //             0 <= tmp_j < input.shape[1] and
                        //             input[tmp_i, tmp_j] == rule['not_stop_by_color']:
                        //         output[tmp_i, tmp_j] = rule['with_color']
                        //         tmp_i += idx_ch[0]
                        //         tmp_j += idx_ch[1]
                        if (input(i, j) == rule.start_by_color) {
                            // tmp_i = i + idx_ch[0]
                            // tmp_j = j + idx_ch[1]
                            int tmp_i = i + idx_ch[0], tmp_j = j + idx_ch[1];

                            // while 0<=tmp_i<input.shape[0] and
                            //     0<=tmp_j<input.shape[1] and
                            //     input[tmp_i, tmp_j] == rule["not_stop_by_color"]:
                            //         output[tmp_i, tmp_j] = rule["with_color"]
                            //         tmp_i += idx_ch[0]
                            //         tmp_j += idx_ch[1]
                            while (0 <= tmp_i and tmp_i < static_cast<int>(input.shape[0]) and
                                   0 <= tmp_j and tmp_j < static_cast<int>(input.shape[1]) and
                                   input(tmp_i, tmp_j) == rule.not_stop_by_color) {
                                output(tmp_i, tmp_j) = rule.with_color;
                                tmp_i += idx_ch[0], tmp_j += idx_ch[1];
                            }
                        }
                    }
                }
            }
        }
    }
    break;

    case DRAW_LINE_TO: {
        for (uint i = 0; i < input.shape[0]; i++) {
            for (uint j = 0; j < input.shape[1]; j++) {
                if (input(i, j) != rule.start_by_color) {
                    continue;
                }

                data_type color = rule.direction_color;
                uint number_0 = count_pixels(output, 0, i, 0, output.shape[1], color);
                uint number_1 = count_pixels(output, i + 1, output.shape[0], 0, output.shape[1], color);
                uint number_2 = count_pixels(output, 0, output.shape[0], 0, j, color);
                uint number_3 = count_pixels(output, 0, output.shape[0], j + 1, output.shape[1], color);

                uint max = std::max(std::max(number_0, number_1), std::max(number_2, number_3));
                Direction direction = RIGHT;

                if (max == number_0) {
                    direction = TOP;
                } else if (max == number_1) {
                    direction = BOTTOM;
                } else if (max == number_2) {
                    direction = LEFT;
                } else {
                    direction = RIGHT;
                }

                // possible_directions: TOP, BOTTOM, LEFT, RIGHT
                static const std::array<std::array<int, 2>, 4> index_change = {{
                    {-1, 0}, {1, 0}, {0, -1}, {0, 1}}};
                uint i_dir = static_cast<uint>(direction);
                assert(i_dir < index_change.size());

                const std::array<int, 2> &idx_ch = index_change[i_dir];
                uint tmp_i = i + idx_ch[0], tmp_j = j + idx_ch[1];
                data_type skip_color = rule.not_stop_by_color_and_skip;

                while (tmp_i < input.shape[0] and tmp_j < input.shape[1]) {
                    // printf("tmp_i=%d tmp_j=%d\n", tmp_i, tmp_j);
                    data_type col = input(tmp_i, tmp_j);

                    if (col != rule.not_stop_by_color and col != rule.not_stop_by_color_and_skip) {
                        break;
                    }

                    if (skip_color == 0 or col != skip_color) {
                        output(tmp_i, tmp_j) = rule.with_color;
                    }

                    tmp_i += idx_ch[0], tmp_j += idx_ch[1];
                }
            }
        }
    }
    break;

    // elif rule["type"] == "distribute_colors":
    case DISTRIBUTE_COLORS: {
        // non_zero_rows = []
        // non_zero_columns = []
        // color_for_row = np.zeros(input.shape[0])
        // color_for_column = np.zeros(input.shape[1])
        std::bitset<MAX_DIMS> non_zero_rows, non_zero_columns;
        static_array<data_type, MAX_DIMS> color_for_row(input.shape[0], 0);
        static_array<data_type, MAX_DIMS> color_for_column(input.shape[1], 0);

        // for i in range(input.shape[0]):
        for (uint i = 0; i < input.shape[0]; i++) {
            // row = input[i]
            static_array<data_type, MAX_DIMS> row = input.get_row(i);

            // colors, counts = np.unique(row, return_counts=True)
            static_array<uint, NUM_COLORS> colors, counts;
            get_unique_with_count(colors, counts, row);

            // good_colors = np.array([c in rule["colors"] for c in colors])
            std::bitset<NUM_COLORS> good_colors = intersection(colors, rule.colors);

            // if not good_colors.any():
            //     continue
            if (!good_colors.any()) {
                continue;
            }

            // colors = colors[good_colors]
            // counts = counts[good_colors]
            // best_color = colors[np.argmax(counts)]
            // color_for_row[i] = best_color
            // non_zero_rows.append(i)
            static_array<uint, NUM_COLORS> indices(colors.size());
            std::iota(indices.begin(), indices.end(), 0);
            int best_idx = *std::max_element(indices.begin(), indices.end(), [&](uint a, uint b) {
                return (good_colors[colors[a]] ? counts[a] : 0) <
                       (good_colors[colors[b]] ? counts[b] : 0); });

            int best_color = colors[best_idx];
            color_for_row[i] = best_color;
            non_zero_rows.set(i);
        }

        // for j in range(input.shape[1]):
        for (uint j = 0; j < input.shape[1]; j++) {
            // row = input[:, j]
            static_array<data_type, MAX_DIMS> col = input.get_col(j);

            // colors, counts = np.unique(row, return_counts=True)
            static_array<uint, NUM_COLORS> colors, counts;
            get_unique_with_count(colors, counts, col);

            // good_colors = np.array([c in rule["colors"] for c in colors])
            std::bitset<NUM_COLORS> good_colors = intersection(colors, rule.colors);

            // if not good_colors.any():
            //     continue
            if (!good_colors.any()) {
                continue;
            }

            // colors = colors[good_colors]
            // counts = counts[good_colors]
            // best_color = colors[np.argmax(counts)]
            // color_for_column[j] = best_color
            // non_zero_columns.append(j)
            static_array<uint, NUM_COLORS> indices(colors.size());
            std::iota(indices.begin(), indices.end(), 0);
            int best_idx = *std::max_element(indices.begin(), indices.end(), [&](uint a, uint b) {
                return (good_colors[colors[a]] ? counts[a] : 0) <
                       (good_colors[colors[b]] ? counts[b] : 0); });

            int best_color = colors[best_idx];
            color_for_column[j] = best_color;
            non_zero_columns.set(j);
        }

        // if rule["horizontally"] == 1:
        //     for i in non_zero_rows:
        //         output[i] = color_for_row[i]
        if (rule.horizontally) {
            for (uint i = 0; i < input.shape[0]; i++) {
                if (non_zero_rows.test(i)) {
                    for (uint j = 0; j < input.shape[1]; j++) {
                        output(i, j) = color_for_row[i];
                    }
                }
            }
        }

        // if rule["vertically"] == 1:
        //     for j in non_zero_columns:
        //         output[:, j] = color_for_column[j]
        if (rule.vertically) {
            for (uint j = 0; j < input.shape[1]; j++) {
                if (non_zero_columns.test(j)) {
                    for (uint i = 0; i < input.shape[0]; i++) {
                        output(i, j) = color_for_column[j];
                    }
                }
            }
        }

        // for i in non_zero_rows:
        //     for j in non_zero_columns:
        //         if input[i, j] == 0:
        //             output[i, j] = rule["intersec"]
        for (uint i = 0; i < input.shape[0]; i++) {
            if (non_zero_rows.test(i)) {
                for (uint j = 0; j < input.shape[1]; j++) {
                    if (non_zero_columns.test(j) and input(i, j) == 0) {
                        output(i, j) = rule.intersect;
                    }
                }
            }
        }
    }
    break;

    // elif rule["type"] == "unity":
    case UNITY: {
        // if rule["mode"] == "vertical":
        if (rule.mode == VERTICAL or rule.mode == HORVER) {
            // for j in range(input.shape[1]):
            for (uint j = 0; j < input.shape[1]; j++) {
                // last_color_now = np.zeros(10, dtype=np.int) - 1
                static_array<int, NUM_COLORS> last_color_now(NUM_COLORS, -1);

                // for i in range(input.shape[0]):
                for (uint i = 0; i < input.shape[0]; i++) {
                    data_type col = input(i, j);

                    // if not input[i, j] in rule["ignore_colors"] and last_color_now[input[i, j]] >= 0:
                    if (!rule.ignore_colors.test(col)) {
                        if (last_color_now[col] >= 0) {
                            // if rule["with_color"] == 0:
                            //     output[(last_color_now[input[i, j]] + 1):i, j] = input[i, j]
                            // else:
                            //     output[(last_color_now[input[i, j]] + 1):i, j] = rule["with_color"]
                            data_type fill_col = (rule.with_color == 0) ? col : rule.with_color;

                            for (uint k = last_color_now[col] + 1; k < i; k++) {
                                output(k, j) = fill_col;
                            }
                        }

                        // last_color_now[input[i, j]] = i
                        last_color_now[col] = i;
                    }
                }
            }
        }

        // elif rule["mode"] == "horizontal":
        if (rule.mode == HORIZONTAL or rule.mode == HORVER) {
            // for i in range(input.shape[0]):
            for (uint i = 0; i < input.shape[0]; i++) {
                // last_color_now = np.zeros(10, dtype=np.int) - 1
                static_array<int, NUM_COLORS> last_color_now(NUM_COLORS, -1);

                // for j in range(input.shape[1]):
                for (uint j = 0; j < input.shape[1]; j++) {
                    data_type col = input(i, j);

                    // if not input[i, j] in rule["ignore_colors"] and last_color_now[input[i, j]] >= 0:
                    if (!rule.ignore_colors.test(col)) {
                        if (last_color_now[col] >= 0) {
                            // if rule["with_color"] == 0:
                            //     output[i, (last_color_now[input[i, j]] + 1):j] = input[i, j]
                            // else:
                            //     output[i, (last_color_now[input[i, j]] + 1):j] = rule["with_color"]
                            data_type fill_col = (rule.with_color == 0) ? col : rule.with_color;

                            for (uint k = last_color_now[col] + 1; k < j; k++) {
                                output(i, k) = fill_col;
                            }
                        }

                        // last_color_now[input[i, j]] = j
                        last_color_now[col] = j;
                    }
                }
            }
        }

        // elif rule["mode"] == "diagonal":
        if (rule.mode == DIAGONAL) {
            for (int flip = 0; flip < 2; flip++) {
                // for diag_id in np.arange(-input.shape[0] - 1, input.shape[1] + 1):
                for (int diag_id = -input.shape[0] - 1; diag_id <= int(input.shape[1]); diag_id++) {
                    // last_color_now_x = np.zeros(10, dtype=np.int) - 1
                    // last_color_now_y = np.zeros(10, dtype=np.int) - 1
                    static_array<int, NUM_COLORS> last_color_now_x(NUM_COLORS, -1);
                    static_array<int, NUM_COLORS> last_color_now_y(NUM_COLORS, -1);

                    // for i, j in zip(np.arange(input.shape[0]), diag_id + np.arange(input.shape[0])):
                    for (uint i = 0; i < input.shape[0]; i++) {
                        uint j = diag_id + i;

                        // if 0 <= i < input.shape[0] and 0 <= j < input.shape[1]:
                        if (j < input.shape[1]) {
                            // if not input[i, j] in rule["ignore_colors"] and last_color_now_x[input[i, j]] >= 0:
                            uint flipped_j = (flip) ? input.shape[1] - j - 1 : j;
                            data_type col = input(i, flipped_j);

                            if (!rule.ignore_colors.test(col)) {
                                if (last_color_now_x[col] >= 0) {
                                    // if rule["with_color"] == 0:
                                    //     output[np.arange(last_color_now_x[input[i, j]]+1, i), np.arange(
                                    //         last_color_now_y[input[i, j]]+1, j)] = input[i, j]
                                    // else:
                                    //     output[np.arange(last_color_now_x[input[i, j]]+1, i), np.arange(
                                    //         last_color_now_y[input[i, j]]+1, j)] = rule[
                                    //         "with_color"]
                                    data_type fill_col = (rule.with_color == 0) ? col : rule.with_color;
                                    int ofs = last_color_now_y[col] - last_color_now_x[col];

                                    for (uint k = last_color_now_x[col] + 1; k < i; k++) {
                                        flipped_j = (flip) ? input.shape[1] - (ofs + k) - 1 : (ofs + k);
                                        output(k, flipped_j) = fill_col;
                                    }
                                }

                                // last_color_now_x[input[i, j]] = i
                                // last_color_now_y[input[i, j]] = j
                                last_color_now_x[col] = i;
                                last_color_now_y[col] = j;
                            }
                        }
                    }
                }
            }

        }
    }
    break;

    // elif rule["type"] == "map_color":
    //     output[output == rule["color_in"]] = rule["color_out"]
    case MAP_COLOR: {
        for (uint i = 0; i < input.shape[0]; i++) {
            for (uint j = 0; j < input.shape[1]; j++) {
                if (output(i, j) == rule.color_in) {
                    output(i, j) = rule.color_out;
                }
            }
        }
    }
    break;

    // elif rule["type"] == "make_holes":
    //     for i in range(output.shape[0]):
    //         for j in range(output.shape[1]):
    //             i_nbh = get_neighbours(output, i, j)
    //             proper_nbhs = i_nbh.values()
    //             for color in np.arange(1, 10):
    //                 if sum(1 for v in proper_nbhs if v == color) == 8:
    //                     output[i, j] = 0
    //                     break
    case MAKE_HOLES: {
        for (uint i = 0; i < input.shape[0]; i++) {
            for (uint j = 0; j < input.shape[1]; j++) {
                NeisArray i_nbh = get_neighbours(output, i, j, NeisAll);

                if (i_nbh.size() == 8 and i_nbh[0] and i_nbh.count(i_nbh[0]) == 8) {
                    output(i, j) = 0;
                }
            }
        }
    }
    break;

    // elif rule["type"] == "gravity":
    case GRAVITY: {
        // changed_smth = 1
        // im = output
        bool changed_smth = true;

        // if rule["gravity_type"] == "figures":
        //     communities = get_graph_communities(im, ignore_black = True)
        //
        // else:
        //     communities = []
        //     for i in range(output.shape[0]):
        //         for j in range(output.shape[1]):
        //             if output[i, j] > 0:
        //                 communities.append([[i, j]])
        std::vector<Island> communities;

        if (rule.gravity_type == FIGURES) {
            communities = get_connectivity_info(input, true);
        } else {
            communities.reserve(input.color.size());

            for (uint i = 0; i < input.shape[0]; i++) {
                for (uint j = 0; j < input.shape[1]; j++) {
                    if (output(i, j) > 0) {
                        communities.push_back({Point{i, j}});
                    }
                }
            }
        }

        // directions = []
        static_array<Direction, MAX_AREA> directions(communities.size());
        std::bitset<MAX_AREA> already_moved;

        // for com in communities:
        for (uint comm_idx = 0; comm_idx < communities.size(); comm_idx++) {
            const Island &com = communities[comm_idx];

            // community = list(com)
            // color_fig = output[community[0][0], community[0][1]]
            data_type color_fig = output(com[0].x, com[0].y);

            // if rule["look_at_what_to_move"] == 1 and color_fig != rule["color_what"]:
            if (rule.look_at_what_to_move and color_fig != rule.color_what) {
                // directions.append("None")
                // continue
                directions[comm_idx] = NONE;
                continue;
            }

            // xs = [p[0] for p in community]
            // ys = [p[1] for p in community]
            // if rule["direction_type"] == "border":
            //     direction = rule["direction_border"]
            Direction direction = NONE;

            if (rule.direction_type == BORDER) {
                direction = rule.direction_border;
            // elif rule["direction_type"] == "color":
            } else if (rule.direction_type == COLOR) {
                // color = rule["direction_color"]
                // xmin, xmax = np.min(xs), np.max(xs)
                // ymin, ymax = np.min(ys), np.max(ys)
                // number_0 = np.sum(output[:xmin] == color)
                // number_1 = np.sum(output[(xmax + 1):] == color)
                // number_2 = np.sum(output[:, :ymin] == color)
                // number_3 = np.sum(output[:, (ymax + 1):] == color)
                // direction = ["up", "down", "left", "right"][np.argmax([number_0, number_1, number_2, number_3])]
                data_type color = rule.direction_color;

                uint xmin = std::min_element(com.begin(), com.end(),
                    [](const Point &a, const Point &b) { return a.x < b.x; })->x;
                uint xmax = std::min_element(com.begin(), com.end(),
                    [](const Point &a, const Point &b) { return a.x > b.x; })->x;
                uint ymin = std::min_element(com.begin(), com.end(),
                    [](const Point &a, const Point &b) { return a.y < b.y; })->y;
                uint ymax = std::min_element(com.begin(), com.end(),
                    [](const Point &a, const Point &b) { return a.y > b.y; })->y;

                uint number_0 = count_pixels(output, 0, xmin, 0, output.shape[1], color);
                uint number_1 = count_pixels(output, xmax+1, output.shape[0], 0, output.shape[1], color);
                uint number_2 = count_pixels(output, 0, output.shape[0], 0, ymin, color);
                uint number_3 = count_pixels(output, 0, output.shape[0], ymax+1, output.shape[1], color);

                uint max = std::max(std::max(number_0, number_1), std::max(number_2, number_3));

                if (max == number_0) {
                    direction = TOP;
                } else if (max == number_1) {
                    direction = BOTTOM;
                } else if (max == number_2) {
                    direction = LEFT;
                } else {
                    direction = RIGHT;
                }
            } else {
                assert(false);
            }

            // directions.append(direction)
            assert(direction < 4);
            directions[comm_idx] = direction;
        }

        // while changed_smth > 0:
        //     changed_smth = 0
        while (changed_smth) {
            changed_smth = false;

            // for i, com in enumerate(communities):
            for (uint comm_idx = 0; comm_idx < communities.size(); comm_idx++) {
                Island &island = communities[comm_idx];
                Direction direction = directions[comm_idx];

                if (direction == NONE) {
                    continue;
                }

                // community = list(com)
                // color_fig = output[community[0][0], community[0][1]]
                // xs = [p[0] for p in community]
                // ys = [p[1] for p in community]
                static const std::array<std::array<int, 2>, 8> offsets = {{
                    {-1, 0}, {1, 0}, {0, -1}, {0, 1}}};

                data_type color_fig = output(island[0].x, island[0].y);

                // direction = directions[i]
                // if direction == 'up':
                //     toper = np.array([[p[0] - 1, p[1]] for p in community if (p[0] - 1, p[1]) not in community])
                //     xs = np.array([p[0] for p in toper])
                //     ys = np.array([p[1] for p in toper])
                //     if np.min(xs) < 0:
                //         continue
                //
                //     if (output[xs, ys] == 0).all():
                //         changed_smth = 1
                //         com_xs = np.array([p[0] for p in community])
                //         com_ys = np.array([p[1] for p in community])
                //         output[com_xs, com_ys] = 0
                //         output[com_xs - 1, com_ys] = color_fig
                //         communities[i] = [(p[0] - 1, p[1]) for p in community]

                PointSet point_set(island, output.shape);
                assert(direction < offsets.size());
                const std::array<int, 2> offset = offsets[direction];
                bool can_move = true;
                static_array<Point, MAX_AREA> new_points(island.size());

                for (uint i = 0; i < island.size(); i++) {
                    const Point &p = island[i];
                    Point q{p.x + offset[0], p.y + offset[1]};

                    if (q.x >= output.shape[0] or q.y >= output.shape[1] or
                        !point_set.has(q) and output(q.x ,q. y)) {
                            can_move = false;
                            break;
                        }

                    new_points[i] = q;
                }

                if (can_move and (rule.steps_limit or !already_moved[comm_idx])) {
                    changed_smth = true;
                    already_moved[comm_idx] = 1;

                    for (const Point &p : island) {
                        output(p.x, p.y) = 0;
                    }

                    for (uint i = 0; i < island.size(); i++) {
                        const Point &p = new_points[i];

                        output(p.x, p.y) = color_fig;
                        island[i] = p;
                    }
                }

            }
        }
    }
    break;

    // elif rule["type"] == "split_by_H":
    case SPLIT_BY_H: {
        // if output.shape[0] >= 2:
        //     part1 = output[:int(np.floor(output.shape[0]/2))]
        //     part2 = output[int(np.ceil(output.shape[0]/2)):]
        //
        //     output = np.zeros_like(part1)
        //     if rule["merge_rule"] == "or":
        //         output[part1>0] = part1[part1>0]
        //         output[part2>0] = part2[part2>0]
        //     elif rule["merge_rule"] == "equal":
        //         idx = np.logical_and(np.logical_and(part1>0, part2>0), part1==part2)
        //         output[idx] = part1[idx]
        //     elif rule["merge_rule"] == "and":
        //         idx = np.logical_and(part1>0, part2>0)
        //         output[idx] = part1[idx]
        //     elif rule["merge_rule"] == "xor":
        //         idx = np.logical_xor(part1>0, part2>0)
        //         output[idx] = part1[idx]
        if (output.shape[0] >= 2) {
            uint border1 = output.shape[0] / 2;
            uint border2 = (output.shape[0] + 1) / 2;

            for (uint i = 0; i < border1; i++) {
                for (uint j = 0; j < output.shape[1]; j++) {
                    data_type col1 = output(i, j), col2 = output(i + border2, j);

                    if (rule.merge_rule == OR) {
                        output(i, j) = (col2 ? col2 : (col1 ? col1 : 0));
                    } else if (rule.merge_rule == EQUAL) {
                        output(i, j) = (col1 == col2) ? col1 : 0;
                    } else if (rule.merge_rule == AND) {
                        output(i, j) = (col1 and col2) ? col1 : 0;
                    } else if (rule.merge_rule == XOR) {
                        output(i, j) = (bool(col1) ^ bool(col2)) ? col1 : 0;
                    } else {
                        assert(false);
                    }
                }
            }

            crop_image(output, 0, border1, 0, output.shape[1]);
        }
    }
    break;

    // elif rule["type"] == "split_by_W":
    case SPLIT_BY_W: {
        // if output.shape[1] >= 2:
        //     part1 = output[:, :int(np.floor(output.shape[1]/2))]
        //     part2 = output[:, int(np.ceil(output.shape[1]/2)):]
        //     output = np.zeros_like(part1)
        //     if rule["merge_rule"] == "or":
        //         output[part1>0] = part1[part1>0]
        //         output[part2>0] = part2[part2>0]
        //     elif rule["merge_rule"] == "equal":
        //         idx = np.logical_and(np.logical_and(part1>0, part2>0), part1==part2)
        //         output[idx] = part1[idx]
        //     elif rule["merge_rule"] == "and":
        //         idx = np.logical_and(part1>0, part2>0)
        //         output[idx] = part1[idx]
        //     elif rule["merge_rule"] == "xor":
        //         idx = np.logical_xor(part1>0, part2>0)
        //         output[idx] = part1[idx]
        if (output.shape[1] >= 2) {
            uint border1 = output.shape[1] / 2;
            uint border2 = (output.shape[1] + 1) / 2;

            for (uint i = 0; i < output.shape[0]; i++) {
                for (uint j = 0; j < border1; j++) {
                    data_type col1 = output(i, j), col2 = output(i, j + border2);

                    if (rule.merge_rule == OR) {
                        output(i, j) = (col2 ? col2 : (col1 ? col1 : 0));
                    } else if (rule.merge_rule == EQUAL) {
                        output(i, j) = (col1 == col2) ? col1 : 0;
                    } else if (rule.merge_rule == AND) {
                        output(i, j) = (col1 and col2) ? col1 : 0;
                    } else if (rule.merge_rule == XOR) {
                        output(i, j) = (bool(col1) ^ bool(col2)) ? col1 : 0;
                    } else {
                        assert(false);
                    }
                }
            }

            crop_image(output, 0, output.shape[0], 0, border1);
        }
    }
    break;

    // elif rule["type"] == "crop_empty":
    case CROP_EMPTY: {
        // nonzerosi = np.max((output != 0).astype(np.int), axis=1)
        // nonzerosj = np.max((output != 0).astype(np.int), axis=0)
        //
        // if np.max(nonzerosi) == 0 or np.max(nonzerosj) == 0:
        //     output = output * 0
        // else:
        //     mini = np.min(np.arange(output.shape[0])[nonzerosi==1])
        //     maxi = np.max(np.arange(output.shape[0])[nonzerosi==1])
        //     minj = np.min(np.arange(output.shape[1])[nonzerosj==1])
        //     maxj = np.max(np.arange(output.shape[1])[nonzerosj==1])
        //     output = output[mini:(maxi+1), minj:(maxj+1)]

        uint xmin = UINT_MAX, ymin = UINT_MAX, xmax = 0, ymax = 0;

        for (uint i = 0; i < output.shape[0]; i++) {
            for (uint j = 0; j < output.shape[1]; j++) {
                if (output(i, j)) {
                    xmin = std::min(xmin, i);
                    xmax = std::max(xmax, i);
                    ymin = std::min(ymin, j);
                    ymax = std::max(ymax, j);
                }
            }
        }

        if (xmax < xmin) {
            output.color.fill(0);
        } else {
            crop_image(output, xmin, xmax + 1, ymin, ymax + 1);
        }
    }
    break;

    // elif rule["type"] == "crop_figure":
    case CROP_FIGURE: {
        // communities = get_graph_communities(output, ignore_black=True)
        // if len(communities) == 0:
        //     output = np.zeros_like(output)
        // else:
        //     if rule["mode"] == "biggest":
        //         biggest = list(communities[np.argmax([len(list(com)) for com in communities])])
        //     else:
        //         biggest = list(communities[np.argmin([len(list(com)) for com in communities])])
        //     biggest = np.array(biggest)
        //     min_bx = np.min(biggest[:, 0])
        //     min_by = np.min(biggest[:, 1])
        //     biggest[:, 0] -= min_bx
        //     biggest[:, 1] -= min_by
        //     output = np.zeros((np.max(biggest[:, 0])+1, np.max(biggest[:, 1])+1), dtype=np.int)
        //     for i in range(biggest.shape[0]):
        //         output[tuple(biggest[i])] = input[(min_bx + biggest[i][0], min_by + biggest[i][1])]

        std::vector<Island> communities = get_connectivity_info(input, true, rule.dif_c_edge);

        if (communities.empty()) {
            output.color.fill(0);
        } else {
            int best = 0;

            for (uint i = 1; i < communities.size(); i++) {
                if (rule.mode == BIGGEST and communities[i].size() > communities[best].size() or
                    rule.mode != BIGGEST and communities[i].size() < communities[best].size()) {
                        best = i;
                }
            }

            const Island &biggest = communities[best];
            uint xmin = UINT_MAX, ymin = UINT_MAX, xmax = 0, ymax = 0;

            for (const Point &p : biggest) {
                xmin = std::min(xmin, p.x);
                xmax = std::max(xmax, p.x);
                ymin = std::min(ymin, p.y);
                ymax = std::max(ymax, p.y);
            }

            PointSet point_set(biggest, output.shape);

            for (uint i = xmin; i <= xmax; i++) {
                for (uint j = ymin; j <= ymax; j++) {
                    if (!point_set.has(Point{i, j})) {
                        output(i, j) = 0;
                    }
                }
            }

            crop_image(output, xmin, xmax + 1, ymin, ymax + 1);
        }
    }
    break;

    default:
        assert(false);
    }

    return output;
}


std::string read_file(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("could not open %s\n", filename);
        exit(1);
    }

    fseek(file, 0, SEEK_END);
    uint length = ftell(file);
    fseek(file, 0, SEEK_SET);

    std::vector<char> buffer(length + 1, 0);
    if (fread(buffer.data(), 1, length, file) != length) {
        printf("file read error %s\n", filename);
        exit(1);
    }

    fclose(file);
    return buffer.data();
}

json11::Json parse_json(const std::string &text) {
    std::string error;
    json11::Json json = json11::Json::parse(text.data(), error, json11::JsonParse::COMMENTS);
    // assert(!json.is_null());
    return json;
}

std::bitset<NUM_COLORS> parse_set(const json11::Json &json) {
    std::bitset<NUM_COLORS>  res;

    for (const json11::Json &item : json.array_items()) {
        assert(item.is_number());
        res.set(item.int_value());
    }

    return res;
}

Rule parse_rule(const json11::Json &json) {
    Rule res{};

    for (const auto &iter : json.object_items()) {
        const std::string &key = iter.first;
        const json11::Json &value = iter.second;

        std::string string_value = value.string_value();
        std::transform(string_value.begin(), string_value.end(), string_value.begin(),
            [](unsigned char c){ return std::tolower(c); });

        // decode enum values
        if (key == "type") {
            assert(value.is_string());
            res.type = static_cast<RuleType>(string2enum.find(string_value)->second);
        } else if (key == "mode") {
            assert(value.is_string());
            res.mode = static_cast<Direction>(string2enum.find(string_value)->second);
        } else if (key == "direction") {
            assert(value.is_string());
            res.direction = static_cast<Direction>(string2enum.find(string_value)->second);
        } else if (key == "gravity_type") {
            assert(value.is_string());
            res.gravity_type = static_cast<RuleType>(string2enum.find(string_value)->second);
        } else if (key == "direction_type") {
            assert(value.is_string());
            res.direction_type = static_cast<Direction>(string2enum.find(string_value)->second);
        } else if (key == "direction_border") {
            assert(value.is_string());
            res.direction_border = static_cast<Direction>(string2enum.find(string_value)->second);
        } else if (key == "merge_rule") {
            assert(value.is_string());
            res.merge_rule = static_cast<MergeRule>(string2enum.find(string_value)->second);
        } else if (key == "apply_to") {
            assert(value.is_string());
            res.apply_to = static_cast<Direction>(string2enum.find(string_value)->second);
        } else if (key == "sort") {
            assert(value.is_string());
            res.sort = static_cast<Direction>(string2enum.find(string_value)->second);
        } else if (key == "how") {
            assert(value.is_string());
            res.how = static_cast<Direction>(string2enum.find(string_value)->second);
        } else if (key == "macro_type") {
            assert(value.is_string());
            res.macro_type = static_cast<MacroType>(string2enum.find(string_value)->second);

        // integer values
        } else if (key == "horizontally") {
            assert(value.is_number());
            res.horizontally = value.int_value();
        } else if (key == "vertically") {
            assert(value.is_number());
            res.vertically = value.int_value();
        } else if (key == "intersect") {
            assert(value.is_number());
            res.intersect = value.int_value();
        } else if (key == "check_in_empty") {
            assert(value.is_number());
            res.check_in_empty = value.int_value();
        } else if (key == "color_in") {
            assert(value.is_number());
            res.color_in = value.int_value();
        } else if (key == "color_out") {
            assert(value.is_number());
            res.color_out = value.int_value();
        } else if (key == "start_by_color") {
            assert(value.is_number());
            res.start_by_color = value.int_value();
        } else if (key == "not_stop_by_color") {
            assert(value.is_number());
            res.not_stop_by_color = value.int_value();
        } else if (key == "nbh_check_out") {
            assert(value.is_number());
            res.nbh_check_out = value.int_value();
        } else if (key == "nbh_check_sum") {
            assert(value.is_number());
            res.nbh_check_sum = value.int_value();
        } else if (key == "with_color") {
            res.with_color = value.int_value();
        } else if (key == "look_at_what_to_move") {
            res.look_at_what_to_move = value.int_value();
        } else if (key == "color_what") {
            res.color_what = value.int_value();
        } else if (key == "direction_color") {
            res.direction_color = value.int_value();
        } else if (key == "steps_limit") {
            res.steps_limit = value.int_value();
        } else if (key == "look_back_color") {
            res.look_back_color = value.int_value();
        } else if (key == "apply_to_index") {
            res.apply_to_index = value.int_value();
        } else if (key == "allow_color") {
            res.allow_color = value.int_value();
        } else if (key == "rotations_count") {
            res.rotations_count = value.int_value();
        } else if (key == "dif_c_edge") {
            res.dif_c_edge = value.int_value();
        } else if (key == "k") {
            res.k = value.int_value();
        } else if (key == "k1") {
            res.k1 = value.int_value();
        } else if (key == "k2") {
            res.k2 = value.int_value();
        } else if (key == "skip_color") {
            res.skip_color = value.int_value();
        } else if (key == "not_stop_by_color_and_skip") {
            res.not_stop_by_color_and_skip = value.int_value();
        } else if (key == "fill_with_color") {
            res.fill_with_color = value.int_value();

        // arrays of integers
        } else if (key == "ignore_colors") {
            res.ignore_colors = parse_set(value);
        } else if (key == "nbh_check_colors") {
            res.nbh_check_colors = parse_set(value);
        } else if (key == "colors") {
            res.colors = parse_set(value);
        } else if (key == "copy_color") {
            res.copy_color = parse_set(value);

        } else {
            printf("unknown parameter: \"%s\"\n", key.c_str());
            assert(false);
        }
    }

    return res;
}

// void test_program(const char *task_path, int sample_idx, const char *program_json) {
//     // parse JSON task
//     json11::Json task = parse_json(read_file(task_path));
//     assert(!task.is_null());
//
//     json11::Json::array train_samples = task["train"].array_items();
//     const json11::Json &task_sample = train_samples[sample_idx];
//
//     AutomatonState state(task_sample["input"]);
//
//
//     // parse the program
//     json11::Json rules_json = parse_json(read_file(program_json));
//     assert(!rules_json.is_null());
//     json11::Json::array rules_arr = rules_json.array_items();
//     assert(rules_arr.size() == 2);
//
//     AutomatonParams params;
//
//     for (const json11::Json &rule : rules_arr[0].array_items()) {
//         assert(rule["macro_type"].string_value() == "global_rule");
//         params.global_rules.push_back(parse_rule(rule));
//     }
//
//     for (const json11::Json &rule : rules_arr[1].array_items()) {
//         assert(rule["macro_type"].string_value() == "ca_rule");
//         params.ca_rules.push_back(parse_rule(rule));
//     }
//
//
//     if (params.global_rules.size() + params.ca_rules.size() > 1) {
//         // run the whole automaton
//         state = trace_param_automata(state, params);
//     } else {
//         // run a single rule
//         if (params.global_rules.size()) {
//             state = apply_rule(state, params.global_rules[0]);
//         } else {
//             state = compute_parametrized_automata(state, params.ca_rules);
//         }
//     }
//
//
//     // output the result
//     state.print();
// }

size_t run_automaton_for_sample(char *buffer, size_t buffer_size, const char *task_str) {
    // parse JSON task
    const json11::Json &task_json = parse_json(task_str);
    if (task_json.is_null()) {
        printf("input is not a valid json: '%s'\n", task_str);
        return 0;
    }

    const json11::Json::object &task = task_json.object_items();

    // json11::Json::array train_samples = task["train"].array_items();
    // const json11::Json &task_sample = train_samples[sample_idx];

    AutomatonState state(task.find("input")->second);

    // parse the program
    const json11::Json &rules_json = task.find("params")->second;
    const json11::Json::array &rules_arr = rules_json.array_items();
    assert(rules_arr.size() == 4);

    AutomatonParams params;

    for (const json11::Json &rule : rules_arr[0].array_items()) {
        // assert(rule.find("macro_type")->second.string_value() == "global_rule");
        params.global_rules.push_back(parse_rule(rule));
    }

    for (const json11::Json &rule : rules_arr[1].array_items()) {
        // assert(rule.find("macro_type")->second.string_value() == "ca_rule");
        params.ca_rules.push_back(parse_rule(rule));
    }

    params.split_rule = parse_rule(rules_arr[2]);
    params.merge_rule = parse_rule(rules_arr[3]);


    // run the whole automaton
    int n_iter = 25; // task.find("n_iter")->second.int_value();
    state = trace_param_automata(state, params, n_iter);


    // output the result
    size_t buffer_pos = 0;

    for (uint i = 0; i < state.shape[0]; i++) {
        for (uint j = 0; j < state.shape[1]; j++) {
            buffer[buffer_pos++] = '0' + state(i, j);
        }

        buffer[buffer_pos++] = '|';
    }

    assert(buffer_pos <= buffer_size);
    buffer[--buffer_pos] = '\0';
    return buffer_pos;
}


#if ENABLE_MAIN

int main(int argc, char *argv[]) {
    std::set_terminate([](){ assert(!"unhandled exception"); });

    // if (argc == 4) {
    //     test_program(argv[1], std::atoi(argv[2]), argv[3]);
    // }
    if (argc == 2) {
        char buf[64 * 1024] = {};
        run_automaton_for_sample(buf, sizeof(buf), read_file(argv[1]).c_str());

        for (char *pos = buf; *pos; pos++) if (*pos == '|') *pos = '\n';
        printf("output:\n%s\n", buf);
    } else {
        printf("usage: %s <test_case.json>\n", argv[0]);
    }
}

#else // use pybind11


int decode_str(const py::handle &val) {
    std::string string_value{val.cast<py::str>()};

    std::transform(string_value.begin(), string_value.end(), string_value.begin(),
        [](unsigned char c){ return std::tolower(c); });

    auto it = string2enum.find(string_value);

    if (it == string2enum.end()) {
        printf("value not found: %s\n", string_value.c_str());
        assert(false);
        return 0;
    }

    return it->second;
}

std::bitset<NUM_COLORS> convert_set(const py::handle &val) {
    std::bitset<NUM_COLORS> res;

    for (py::handle item : val.cast<py::list>()) {
        res.set(item.cast<int>());
    }

    return res;
}

Rule convert_rule(const py::dict &dict) {
    Rule res{};

    for (const auto &iter : dict) {
        const std::string &key = iter.first.cast<py::str>();
        const py::handle &value = iter.second;

        // decode enum values
        if (key == "type") {
            res.type = static_cast<RuleType>(decode_str(value));
        } else if (key == "mode") {
            res.mode = static_cast<Direction>(decode_str(value));
        } else if (key == "direction") {
            res.direction = static_cast<Direction>(decode_str(value));
        } else if (key == "gravity_type") {
            res.gravity_type = static_cast<RuleType>(decode_str(value));
        } else if (key == "direction_type") {
            res.direction_type = static_cast<Direction>(decode_str(value));
        } else if (key == "direction_border") {
            res.direction_border = static_cast<Direction>(decode_str(value));
        } else if (key == "merge_rule") {
            res.merge_rule = static_cast<MergeRule>(decode_str(value));
        } else if (key == "apply_to") {
            res.apply_to = static_cast<Direction>(decode_str(value));
        } else if (key == "sort") {
            res.sort = static_cast<Direction>(decode_str(value));
        } else if (key == "how") {
            res.how = static_cast<Direction>(decode_str(value));
        } else if (key == "macro_type") {
            res.macro_type = static_cast<MacroType>(decode_str(value));

        // integer values
        } else if (key == "horizontally") {
            res.horizontally = value.cast<int>();
        } else if (key == "vertically") {
            res.vertically = value.cast<int>();
        } else if (key == "intersect") {
            res.intersect = value.cast<int>();
        } else if (key == "check_in_empty") {
            res.check_in_empty = value.cast<int>();
        } else if (key == "color_in") {
            res.color_in = value.cast<int>();
        } else if (key == "color_out") {
            res.color_out = value.cast<int>();
        } else if (key == "start_by_color") {
            res.start_by_color = value.cast<int>();
        } else if (key == "not_stop_by_color") {
            res.not_stop_by_color = value.cast<int>();
        } else if (key == "nbh_check_out") {
            res.nbh_check_out = value.cast<int>();
        } else if (key == "nbh_check_sum") {
            res.nbh_check_sum = value.cast<int>();
        } else if (key == "with_color") {
            res.with_color = value.cast<int>();
        } else if (key == "look_at_what_to_move") {
            res.look_at_what_to_move = value.cast<int>();
        } else if (key == "color_what") {
            res.color_what = value.cast<int>();
        } else if (key == "direction_color") {
            res.direction_color = value.cast<int>();
        } else if (key == "steps_limit") {
            res.steps_limit = value.cast<int>();
        } else if (key == "look_back_color") {
            res.look_back_color = value.cast<int>();
        } else if (key == "apply_to_index") {
            res.apply_to_index = value.cast<int>();
        } else if (key == "allow_color") {
            res.allow_color = value.cast<int>();
        } else if (key == "rotations_count") {
            res.rotations_count = value.cast<int>();
        } else if (key == "dif_c_edge") {
            res.dif_c_edge = value.cast<int>();
        } else if (key == "k") {
            res.k = value.cast<int>();
        } else if (key == "k1") {
            res.k1 = value.cast<int>();
        } else if (key == "k2") {
            res.k2 = value.cast<int>();
        } else if (key == "skip_color") {
            res.skip_color = value.cast<int>();
        } else if (key == "not_stop_by_color_and_skip") {
            res.not_stop_by_color_and_skip = value.cast<int>();
        } else if (key == "fill_with_color") {
            res.fill_with_color = value.cast<int>();

        // arrays of integers
        } else if (key == "ignore_colors") {
            res.ignore_colors = convert_set(value);
        } else if (key == "nbh_check_colors") {
            res.nbh_check_colors = convert_set(value);
        } else if (key == "colors") {
            res.colors = convert_set(value);
        } else if (key == "copy_color") {
            res.copy_color = convert_set(value);

        } else {
            printf("unknown parameter: \"%s\"\n", key.c_str());
            assert(false);
        }
    }

    return res;
}

void convert_all_rules(AutomatonParams &params, const py::list &params_list) {
    assert(params_list.size() == 4);

    for (const auto &item : params_list[0].cast<py::list>()) {
        params.global_rules.push_back(convert_rule(item.cast<py::dict>()));
    }

    for (const auto &item : params_list[1].cast<py::list>()) {
        params.ca_rules.push_back(convert_rule(item.cast<py::dict>()));
    }

    params.split_rule = convert_rule(params_list[2].cast<py::dict>());
    params.merge_rule = convert_rule(params_list[3].cast<py::dict>());
}

void convert_automaton_state(AutomatonState &state, const py::array_t<ssize_t> &input) {
    auto buf = input.request();
    char *ptr = (char *) buf.ptr;
    ssize_t stride0 = input.strides(0), stride1 = input.strides(1);

    for (uint i = 0; i < state.shape[0]; i++) {
        for (uint j = 0; j < state.shape[1]; j++) {
            state(i, j) = (ssize_t &) ptr[i * stride0 + j * stride1];
        }
    }
}

py::array_t<ssize_t> convert_result(const AutomatonState &state) {
    py::array_t<ssize_t> output(state.color.size());
    auto buf = output.request();
    ssize_t *ptr = (ssize_t *) buf.ptr;

    for (uint i = 0; i < state.shape[0]; i++) {
        for (uint j = 0; j < state.shape[1]; j++) {
            ptr[i * state.shape[1] + j] = state(i, j);
        }
    }

    output.resize({state.shape[0], state.shape[1]});
    // printf("output shape: %d x %d\n", (int) output.shape(0), (int) output.shape(1));
    return output;
}

py::array_t<ssize_t> cpp_trace_param_automata(py::array_t<ssize_t> input, py::list params_list,
                                              int n_iter) {
    // printf("params_list: %d\n", (int) params_list.size());
    // printf("ndims: %d\n", (int) input.ndim());
    // printf("shape: %d x %d\n", (int) input.shape(0), (int) input.shape(1));
    // printf("strides: %d, %d\n", (int) input.strides(0), (int) input.strides(1));

    // puts(std::string(input.cast<py::str>()).c_str());
    // puts(std::string(params_list.cast<py::str>()).c_str());

    assert(params_list.ndim() == 4);
    assert(input.ndim() == 2);

    AutomatonState state{static_cast<uint>(input.shape()[0]), static_cast<uint>(input.shape()[1])};
    AutomatonParams params;

    convert_automaton_state(state, input);
    convert_all_rules(params, params_list);

    // run the automaton
    state = trace_param_automata(state, params, n_iter);

    return convert_result(state);
}


PYBIND11_MODULE(dsl, m) {
    m.def("cpp_trace_param_automata", &cpp_trace_param_automata,
          "A function which simulates an automaton");
}

#endif
