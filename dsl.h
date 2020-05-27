
#pragma once


using data_type = char;

class AutomatonState {
public:
    using storage_type = static_array<data_type, MAX_DIMS * MAX_DIMS>;

    storage_type color;
    std::array<uint, 2> shape;


    AutomatonState(uint i, uint j) : shape({i, j}) {
        color.resize(shape[0] * shape[1]);
        color.fill(0);
    }

    explicit AutomatonState(const std::array<uint, 2> &shape) : shape(shape) {
        color.resize(shape[0] * shape[1]);
        color.fill(0);
    }

    explicit AutomatonState(const json11::Json &input) {
        assert(input.is_array());

        const json11::Json::array &rows = input.array_items();
        const json11::Json::array &col0 = rows[0].array_items();

        uint height = rows.size(), width = col0.size();
        shape[0] = height, shape[1] = width;

        color.resize(width * height);

        for (uint i = 0; i < height; i++) {
            assert(rows[i].is_array());

            const json11::Json::array &row = rows[i].array_items();
            assert(row.size() == width);

            for (uint j = 0; j < width; j++) {
                const json11::Json &val = row[j];
                color[i * width + j] = val.int_value();
            }
        }
    }

    data_type &operator()(uint i, uint j) {
        return color[i * shape[1] + j];
    }

    data_type operator()(uint i, uint j) const {
        return color[i * shape[1] + j];
    }


    static_array<data_type, MAX_DIMS> get_row(uint i) const {
        assert(i < shape[0]);
        static_array<data_type, MAX_DIMS> res(shape[1]);

        for (uint j = 0; j < shape[1]; j++) {
            res[j] = color[i * shape[1] + j];
        }

        return res;
    }

    static_array<data_type, MAX_DIMS> get_col(uint j) const {
        assert(j < shape[1]);
        static_array<data_type, MAX_DIMS> res(shape[0]);

        for (uint i = 0; i < shape[0]; i++) {
            res[i] = color[i * shape[1] + j];
        }

        return res;
    }


    AutomatonState remove_rows(const std::bitset<MAX_DIMS> &rows) const {
        AutomatonState res{shape[0] - static_cast<uint>(rows.count()), shape[1]};

        for (uint i = 0, k = 0; i < shape[0]; i++) {
            if (!rows[i]) {
                for (uint j = 0; j < shape[1]; j++) {
                    res(k, j) = (*this)(i, j);
                }

                k++;
            }
        }

        return res;
    }

    AutomatonState remove_cols(const std::bitset<MAX_DIMS> &cols) const {
        AutomatonState res{shape[0], shape[1] - static_cast<uint>(cols.count())};

        for (uint j = 0, k = 0; j < shape[1]; j++) {
            if (!cols[j]) {
                for (uint i = 0; i < shape[0]; i++) {
                    res(i, k) = (*this)(i, j);
                }

                k++;
            }
        }

        return res;
    }


    bool operator ==(const AutomatonState &another) const {
        return shape == another.shape && color == another.color;
    }

    void print() const {
        print_image(color, shape[0], shape[1]);
    }
};
