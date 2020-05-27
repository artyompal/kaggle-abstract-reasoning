//
// Some useful algorithms.
//

#pragma once


template <typename T>
void print_image(const T &data, int height, int width)
{
    for (int y = 0; y < height; y++) {
        putchar('|');

        for (int x = 0; x < width; x++) {
            printf("%d", data[y * width + x]);
        }

        puts("|");
    }
}

template <typename T>
void print_image_diff(const T &data, const T &another, int height, int width)
{
    for (int y = 0; y < height; y++) {
        putchar('|');

        for (int x = 0; x < width; x++) {
            if (data[y * width + x] != another[y * width + x]) {
                printf("%d", data[y * width + x]);
            } else {
                putchar('-');
            }
        }

        puts("|");
    }
}

template <typename T>
void print_array(const T &vec)
{
    fprintf(stderr, "[ ");

    for (const auto x : vec) {
        fprintf(stderr, "%d ", x);
    }

    fprintf(stderr, "]\n");
}

// template <typename T>
// void print_array(const T<float> &vec)
// {
//     printf("[ ");
//     for (const auto x : vec) printf("%f ", x);
//     puts("]");
// }

template <unsigned long U>
void print_bitset(const std::bitset<U> &set)
{
    fprintf(stderr, "[ ");

    for (uint i = 0; i < set.size(); i++) {
        if (set.test(i)) {
            fprintf(stderr, "%d ", i);
        }
    }

    fprintf(stderr, "]\n");
}


template<typename T>
void update_maximum(std::atomic<T> &maximum_value, const T &value) {
    T prev_value = maximum_value;
    while (prev_value < value && !maximum_value.compare_exchange_weak(prev_value, value)) {
    }
}

template<typename T, typename U>
uint intersection_size(const T &values, const std::unordered_set<U> &table) {
    int count = 0;

    for (const auto &v : values) {
        count += table.count(v);
    }

    return count;
}

template<typename T, long unsigned U>
uint intersection_size(const T &values, const std::bitset<U> &table) {
    int count = 0;

    for (auto val : values) {
        count += table.test(val);
    }

    return count;
}

template<typename T, long unsigned U>
std::bitset<U> intersection(const T &values, const std::bitset<U> &table) {
    std::bitset<U> res;

    for (auto val : values) {
        res.set(val, table.test(val));
    }

    return res;
}

// template <typename T, long unsigned U>
// T filter(const T &values, const std::bitset<U> &mask) {
//     T res;
//
//     for (auto val : values) {
//         if (mask.test(val)) {
//             res.push_back(val);
//         }
//     }
//
//     return res;
// }

template<typename T, typename U>
void get_unique_with_count(static_array<T, NUM_COLORS> &colors,
        static_array<uint, NUM_COLORS> &counts, const U &values) {
    static_array<T, NUM_COLORS> freq(NUM_COLORS, 0);

    for (uint i = 0; i < values.size(); i++) {
        freq[values[i]]++;
    }

    colors.resize(NUM_COLORS);
    counts.resize(NUM_COLORS);

    for (uint i = 0; i < NUM_COLORS; i++) {
        colors[i] = i;
    }

    std::sort(colors.begin(), colors.end(), [&](int a, int b) { return freq[a] > freq[b]; });

    for (uint i = 0; i < colors.size(); i++) {
        counts[i] = freq[colors[i]];
    }

    int num_non_zeros = std::find(counts.begin(), counts.end(), 0) - counts.begin();

    colors.resize(num_non_zeros);
    counts.resize(num_non_zeros);
}

template<typename T>
void get_bounding_box(uint &minrow, uint &mincol, uint &maxrow, uint &maxcol, const T &input) {
    uint min_i = UINT_MAX, min_j = UINT_MAX, max_i = 0, max_j = 0;

    for (uint i = 0; i < input.shape[0]; i++) {
        for (uint j = 0; j < input.shape[1]; j++) {
            if (input(i, j) > 0) {
                min_i = std::min(min_i, i);
                max_i = std::max(max_i, i + 1);
                min_j = std::min(min_j, j);
                max_j = std::max(max_j, j + 1);
            }
        }
    }

    minrow = min_i, mincol = min_j, maxrow = max_i, maxcol = max_j;
}
