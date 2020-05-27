//
// Static array class.
//

#pragma once


template <typename T, uint max_size>
class static_array {
    T       data_[max_size];
    uint    size_;

public:
    explicit static_array(int sz = 0) {
        resize(sz);
    }

    static_array(int sz, T val) {
        resize(sz);
        fill(val);
    }

    static_array(const static_array<T, max_size> &another) {
        resize(another.size_);

        for (uint i = 0; i < size_; i++) {
            data_[i] = another.data_[i];
        }
    }

    void fill(T val = 0) {
        for (uint i = 0; i < size_; i++) {
            data_[i] = val;
        }
    }

    const static_array<T, max_size> &operator =(const static_array<T, max_size> &another) {
        resize(another.size_);

        for (uint i = 0; i < size_; i++) {
            data_[i] = another.data_[i];
        }

        return *this;
    }

    T *begin() { return data_; }
    T *end() { return data_ + size_; }

    const T *begin() const { return data_; }
    const T *end() const { return data_ + size_; }

    T *data() { return data_; }
    const T *data() const { return data_; }
    uint size() const { return size_; }
    bool empty() const { return !size_; }

    const T &operator[](uint idx) const {
        assert(idx < size_);
        return data_[idx];
    }

    T &operator[](uint idx) {
        assert(idx < size_);
        return data_[idx];
    }

    void resize(uint new_size) {
        assert(new_size <= max_size);
        size_ = new_size;
    }

    template <uint another_size>
    bool operator ==(const static_array<T, another_size> &another) const {
        bool res = true;

        for (uint i = 0; i < size_; i++) {
            res &= data_[i] == another.data_[i];
        }

        return res;
    }

    bool all_equal_to(T val) const {
        bool res = true;

        for (uint i = 0; i < size_; i++) {
            res &= data_[i] == val;
        }

        return res;
    }

    uint count(T val) const {
        uint res = 0;

        for (uint i = 0; i < size_; i++) {
            res += data_[i] == val;
        }

        return res;
    }

    uint count(const std::function<bool(T)> &lambda) const {
        uint res = 0;

        for (uint i = 0; i < size_; i++) {
            res += lambda(data_[i]);
        }

        return res;
    }

    template <uint another_size>
    bool operator !=(const static_array<T, another_size> &another) const {
        return !(*this == another);
    }

    void push_back(T val) {
        uint sz = size();
        resize(sz + 1);
        data_[sz] = val;
    }

    T pop_back() {
        T res = back();
        resize(size() - 1);
        return res;
    }

    T front() const {
        return (*this)[0];
    }

    T back() const {
        return (*this)[size_ - 1];
    }
};
