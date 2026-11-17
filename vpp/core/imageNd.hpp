#ifndef VPP_IMAGENd_HPP__
# define VPP_IMAGENd_HPP__

# include <iostream>
# include <algorithm>
# include <iod/sio.hh>
# include <vpp/core/imageNd.hh>
# include <vpp/core/boxNd.hh>
# include <vpp/core/symbols.hh>

# ifndef VPP_DEFAULT_IMAGE_ALIGNMENT

# ifdef __AVX2__
#  define VPP_DEFAULT_IMAGE_ALIGNMENT 32
# else
#  define VPP_DEFAULT_IMAGE_ALIGNMENT 16
# endif

# endif

namespace vpp
{

  template <typename V, unsigned N>
  imageNd<V, N>::imageNd()
  {
  }

  template <typename V, unsigned N>
  template <typename... O>
  imageNd<V, N>::imageNd(const std::initializer_list<int>& dims, const O&... options)
  {
    allocate(dims, iod::D(options...));
    index_rows();
  }

  template <typename V, unsigned N>
  template <typename... O>
  imageNd<V, N>::imageNd(const std::vector<int>& dims, const O&... options)
  {
    allocate(dims, iod::D(options...));
    index_rows();
  }


  template <typename V, unsigned N>
  template <typename... O>
  imageNd<V, N>::imageNd(int ncols, const O&... options)
    : imageNd(make_box1d(ncols), options...)
  {
    static_assert(N == 1, "ImageNd constructor: bad dimension.");
  }

  template <typename V, unsigned N>
  template <typename... O>
  imageNd<V, N>::imageNd(int nrows, int ncols, const O&... options)
    : imageNd(make_box2d(nrows, ncols), options...)
  {
    static_assert(N == 2, "ImageNd constructor: bad dimension.");
  }

  template <typename V, unsigned N>
  template <typename... O>
  imageNd<V, N>::imageNd(int nslices, int nrows, int ncols, const O&... options)
    : imageNd(make_box3d(nslices, nrows, ncols), options...)
  {
    static_assert(N == 3, "ImageNd constructor: bad dimension.");
  }

  template <typename V, unsigned N>
  imageNd<V, N>::imageNd(imageNd<V, N>&& other)
    : ptr_(std::move(other.ptr_))
  {
  }


  template <typename V, unsigned N>
  imageNd<V, N>& imageNd<V, N>::operator=(const imageNd<V, N>& other)
  {
    ptr_ = other.ptr_;
    return *this;
  }

  template <typename V, unsigned N>
  imageNd<V, N>::imageNd(const imageNd<V, N>& other)
  {
    *this = other;
  }


  template <typename V, unsigned N>
  imageNd<V, N>& imageNd<V, N>::operator=(imageNd<V, N>&& other)
  {
    ptr_ = std::move(other.ptr_);
    return *this;
  }

  template <typename V, unsigned N>
  template <typename... O>
  imageNd<V, N>::imageNd(const boxNd<N>& domain, const O&... _options)
  {
    std::vector<int> dims(N);

    for (int i = 0; i < int(N); i++)
      dims[i] = domain.size(i);

    const auto options = iod::D(_options...);

    static_assert(
      !options.has(_data) or options.has(_pitch),
      "You must provide the pitch (Number of bits between the begining of to successive lines when providing a data pointer the image constructor.");

    if (options.has(_data))
      adapt_external_data(domain,
                          (V*) options.get(_data, (V*)0),
                          options.get(_pitch, 0),
                          options.get(_border, 0));
    else
      allocate(dims, options);

    index_rows();
  }

  template <typename V, unsigned N>
  imageNd<V, N>::~imageNd()
  {
    // The shared_ptr this_->ptr_ will destroy the data if needed.
  }

  template <typename V, unsigned N>
  template <typename... O>
  void imageNd<V, N>::allocate(const std::vector<int>& dims, const iod::sio<O...>& options)
  {
    const int align_size = options.get(_aligned, VPP_DEFAULT_IMAGE_ALIGNMENT); // Memory alignment of rows.
    assert(align_size != 0);
    typedef unsigned long long ULL;
    ptr_ = std::make_shared<imageNd_data<V, N>>();
    auto& d = *ptr_;
    d.border_ = options.get(_border, 0);
    d.alignment_ = align_size;

    int border_size = d.border_ * sizeof(V);
    int border_padding = 0;
    if (border_size % align_size)
    {
      border_padding = align_size - (border_size % align_size);
      border_size += border_padding;
    }

    d.pitch_ = dims[N - 1] * sizeof(V) + border_size * 2;
    if (d.pitch_ % align_size) d.pitch_ += align_size - (d.pitch_ % align_size);

    int size = 1;
    for (int i = 0; i < N - 1; i++)
      size *= dims[i] + (d.border_ * 2);
    size *= d.pitch_;

    d.data_ = (V*) malloc(align_size + size);
    d.data_sptr_ = std::shared_ptr<void>(d.data_, [] (V* p) { 
        free((char*)p); 
      });

    if (ULL(d.data_) % align_size)
      d.data_ = (V*)((char*)(d.data_) + align_size - int(ULL(d.data_) % align_size));

    d.data_end_ = d.data_ + size / sizeof(V);
    assert(!(ULL(d.data_) % align_size));
    vint<N> p2;
    for (unsigned i = 0; i < N; i++)
      p2[i] = dims[i] - 1;
    d.domain_ = boxNd<N, int>(vint<N>::Zero(), p2);
    // The buffer holds the border ring around the domain: the slice
    // strides used by coords_to_offset() must match the allocation
    // layout computed above (dims[i] + 2 * border rows per slice).
    d.buffer_domain_ = d.domain_ + vpp::border(d.border_);

    // Set begin_, the address of the first pixel.
    vint<N> p = vint<N>::Ones() * d.border_;
    d.begin_ = (V*)((char*)d.data_ + border_padding + coords_to_offset(p));
  }

  template <typename V, unsigned N>
  void imageNd<V, N>::adapt_external_data(const boxNd<N>& domain, V* data, int pitch, int border)
  {
    ptr_ = std::shared_ptr<imageNd_data<V, N>>(new imageNd_data<V, N>());
    auto& d = *ptr_;

    d.begin_ = data; // Address of the pixel at domain().p1().
    d.pitch_ = pitch; // Signed byte distance between two consecutive rows.
    d.domain_ = domain;
    d.buffer_domain_ = domain;
    d.border_ = border;

    // Alignment of the rows: largest power of two dividing both the
    // data address and the absolute pitch, capped to a sane value so
    // it can safely feed back into allocate() (e.g. through clone()).
    {
      typedef unsigned long long ULL;
      ULL bits = ULL(data);
      if (pitch != 0)
        bits |= ULL(pitch < 0 ? -(long long) pitch : (long long) pitch);
      unsigned int al = 1;
      while (al < 4096 && (bits % (al * 2)) == 0) al *= 2;
      d.alignment_ = al;
    }

    // Addressable range. Rows are located at begin_ + r * pitch, so
    // with a negative pitch increasing row indices move towards lower
    // addresses. Rows span [-border, nrows - 1 + border] and columns
    // span [-border, ncols - 1 + border]: [data_, data_end_) covers
    // exactly domain_with_border(), whichever side of begin_ the rows
    // grow towards.
    long long nrows = 1;
    for (int n = 0; n < N - 1; n++)
      nrows *= domain.size(n);

    long long first_row = -border;
    long long last_row = nrows - 1 + border;
    long long first_off = first_row * (long long) pitch;
    long long last_off = last_row * (long long) pitch;
    char* rows_lo = (char*) data + std::min(first_off, last_off);
    char* rows_hi = (char*) data + std::max(first_off, last_off);

    char* lo = rows_lo - border * (long long) sizeof(V);
    char* hi = rows_hi + (long long) (domain.size(N - 1) - 1 + border) * sizeof(V) + sizeof(V);

    d.data_ = (V*) lo;
    d.data_end_ = (V*) hi;
  }


  template <typename V, unsigned N>
  void imageNd<V, N>::index_rows()
  {
    
    std::vector<V*>& rows = ptr_->rows_;
    rows.clear();
    boxNd<N - 1> row_domain(domain_with_border().p1().template segment<N - 1>(0),
                            domain_with_border().p2().template segment<N - 1>(0));

    int ras = 0;
    for(vint<N - 1> r : row_domain)
    {
      vint<N> p = domain().p1();
      p.template segment<N - 1>(0) = r;

      rows.push_back(this->address_of(p));

      if (p.template segment<N - 1>(0) == (vint<N - 1>::Zero()))
        ras = rows.size() - 1;
    }

    ptr_->rows_array_start_ = &rows[ras];
  }
  
  template <typename V, unsigned N>
  int imageNd<V, N>::coords_to_offset(const vint<N>& p) const
  {
    int row_idx = (N >= 2) ? p[N - 2] : 0;
    int ds = 1;
    for (int i = N - 3; i >= 0; i--)
    {
      ds *= ptr_->buffer_domain_.size(i + 1);
      row_idx += ds * p[i];
    }
    return row_idx * ptr_->pitch_ + p[N - 1] * sizeof(V);
  }

  template <typename V, unsigned N>
  V&
  imageNd<V, N>::operator()(const vint<N>& p)
  {
    assert(domain_with_border().has(p));
    assert(address_of(p) >= (V*) ptr_->data_ and address_of(p) < ptr_->data_end_);
    if (N == 2)
      return (*this)[p[0]][p[1]];
    else
    {
      V* addr = (V*)((char*)ptr_->begin_ + coords_to_offset(p));
      return *addr;
    }
  }

  template <typename V, unsigned N>
  const V&
  imageNd<V, N>::operator()(const vint<N>& p) const
  {
    assert(domain_with_border().has(p));
    assert(address_of(p) >= (const V*) ptr_->data_ and address_of(p) < (const V*) ptr_->data_end_);
    if (N == 2)
      return (*this)[p[0]][p[1]];
    else
    {
      const V* addr = (const V*)((const char*)ptr_->begin_ + coords_to_offset(p));
      return *addr;
    }
  }

  template <typename V, unsigned N>
  V*&
  imageNd<V, N>::operator[](int r)
  {
    return ptr_->rows_array_start_[r];
  }

  template <typename V, unsigned N>
  V* const &
  imageNd<V, N>::operator[](int r) const
  {
    return ptr_->rows_array_start_[r];
  }
  
  template <typename V, unsigned N>
  V
  imageNd<V, N>::linear_interpolate(const vfloat<N>& p) const
  {
    static_assert(N == 2, "linear_interpolate only supports 2d images.");

    vint2 x = p.template cast<int>();
    float a0 = p[0] - x[0];
    float a1 = p[1] - x[1];

    const V* l1 = address_of(x);
    const V* l2 = (const V*)((const char*)l1 + ptr_->pitch_);

    //typedef plus_promotion<V> S;
    typedef cast_to_float<V> S;

    return vpp::cast<V>((1 - a0) * (1 - a1) *  vpp::cast<S>(l1[0]) +
                        a0 * (1 - a1) *  vpp::cast<S>(l2[0]) +
                        (1 - a0) * a1 *  vpp::cast<S>(l1[1]) +
                        a0 * a1 *  vpp::cast<S>(l2[1]));
  }

  template <typename V, unsigned N>
  V*
  imageNd<V, N>::address_of(const vint<N>& p)
  {
    return (V*)((char*)(ptr_->begin_) + coords_to_offset(p));
  }

  template <typename V, unsigned N>
  const V*
  imageNd<V, N>::address_of(const vint<N>& p) const
  {
    return (V*)((char*)(ptr_->begin_) + coords_to_offset(p));
  }

  template <typename V, unsigned N>
  int
  imageNd<V, N>::offset_of(const vint<N>& p) const
  {
    return coords_to_offset(p);
  }

  template <typename V, unsigned N>
  imageNd<V, N>
  imageNd<V, N>::subimage(const boxNd<N>& d)
  {
    imageNd<V, N> res;
    res.ptr_ = std::shared_ptr<imageNd_data<V, N>>(new imageNd_data<V, N>());
    *res.ptr_.get() = *(this->ptr_.get()); // Copy the image data.
    res.ptr_->begin_ = address_of(d.p1());
    res.ptr_->domain_ = boxNd<N>(d.p1() - d.p1(), d.p2() - d.p1());

    // Shift the row pointers to the first column of the subimage.
    const int col_shift = d.p1()[N - 1] - this->domain().p1()[N - 1];
    for (auto& r_start : res.ptr_->rows_)
      r_start += col_shift;

    // Shift the row array so that row 0 of the subimage points to the
    // first row of d in the parent buffer. rows_ is indexed by the
    // flattened coordinates of the first N-1 dimensions.
    int row_shift = 0;
    {
      int ds = 1;
      for (int i = N - 2; i >= 0; i--)
      {
        row_shift += ds * d.p1()[i];
        ds *= this->domain_with_border().size(i);
      }
    }

    res.ptr_->rows_array_start_ = &res.ptr_->rows_.front() +
      (this->ptr_->rows_array_start_ - &(this->ptr_->rows_.front()))
      + row_shift;

    return res;
  }

  template <typename V, unsigned N>
  const imageNd<V, N>
  imageNd<V, N>::const_subimage(const boxNd<N>& d) const
  {
    imageNd<V, N> res;

    res.ptr_ = std::shared_ptr<imageNd_data<V, N>>(new imageNd_data<V, N>());
    *res.ptr_.get() = *(this->ptr_.get());
    res.ptr_->begin_ = const_cast<V*>(address_of(d.p1()));
    res.ptr_->domain_ = boxNd<N>(d.p1() - d.p1(), d.p2() - d.p1());

    // See subimage().
    const int col_shift = d.p1()[N - 1] - this->domain().p1()[N - 1];
    for (auto& r_start : res.ptr_->rows_)
      r_start += col_shift;

    int row_shift = 0;
    {
      int ds = 1;
      for (int i = N - 2; i >= 0; i--)
      {
        row_shift += ds * d.p1()[i];
        ds *= this->domain_with_border().size(i);
      }
    }

    res.ptr_->rows_array_start_ = &res.ptr_->rows_.front() +
      (this->ptr_->rows_array_start_ - &(this->ptr_->rows_.front()))
      + row_shift;

    return res;
  }

};

#endif
