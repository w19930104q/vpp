// Bounds-oriented tests for imageNd: external stride adaptation vs
// internal allocation, negative pitch, odd widths, border access.
//
// Covers clone / slice / convert on fixed image matrices and exports
// address strides (STRIDE lines) and pixel hashes (HASH lines) on
// stdout so runs can be diffed across versions and builds.
//
// Build and run with -fsanitize=address to validate boundedness.
//
// Define VPP_HEAD_CROSSCHECK to skip the guarantees that did not hold
// at 8413773 (negative-pitch ranges, exact external ranges, 3D slice
// row array), so the same file can export reference hashes when
// compiled against the pre-fix headers.

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include <vpp/vpp.hh>

using namespace vpp;

// FNV-1a 64-bit hash of the pixel values of the domain, in row-major
// coordinate order. Independent of the memory layout (pitch, border,
// alignment), so identical logical images hash identically.
template <typename I>
std::uint64_t pixel_hash(const I& img)
{
  std::uint64_t h = 1469598103934665603ull;
  for (auto p : img.domain())
  {
    typename I::value_type v = img(p);
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&v);
    for (size_t i = 0; i < sizeof(v); i++)
    {
      h ^= b[i];
      h *= 1099511628211ull;
    }
  }
  return h;
}

template <typename I>
void export_stride(const char* name, const I& img)
{
  long begin_off = (const char*) img.address_of(img.domain().p1()) - (const char*) img.data();
  std::cout << "STRIDE " << name
            << " pitch=" << img.pitch()
            << " align=" << img.alignment()
            << " border=" << img.border()
            << " begin_off=" << begin_off
            << " range=" << ((const char*) img.data_end() - (const char*) img.data())
            << std::endl;
}

template <typename I>
void export_hash(const char* name, const I& img)
{
  std::cout << "HASH " << name << " ";
  std::cout.width(16);
  std::cout.fill('0');
  std::cout << std::hex << pixel_hash(img) << std::dec << std::endl;
}

// Fixed deterministic matrix pattern.
inline int pattern(int r, int c)
{
  return (r * 7 + c * 13 + r * c * 3 + 5) % 251 - 100;
}

template <typename I>
void fill_pattern(I& img)
{
  pixel_wise(img.domain(), img) | [] (auto p, auto& v) {
    v = pattern(p[0], p[1]);
  };
}

int main()
{
  // ---------------------------------------------------------------
  // 1. Owned odd-width image with border: clone and slice.
  // ---------------------------------------------------------------
  {
    image2d<int> img(7, 5, _border = 2, _aligned = 16);
    fill_pattern(img);
    fill_border_with_value(img, -7);

    export_stride("owned_7x5_b2", img);
    export_hash("owned_7x5_b2", img);

    // Border ring is inside the allocation (ASAN checks the writes).
    pixel_wise(img.domain_with_border(), img) | [] (vint2, int& v) { v = v + 0; };

    // Clone: same logical content, same hash.
    auto cl = clone(img);
    export_stride("clone_7x5_b2", cl);
    export_hash("clone_7x5_b2", cl);
    assert(pixel_hash(cl) == pixel_hash(img));
    for (auto p : img.domain())
      assert(cl(p) == img(p));

    // Slice touching the right and bottom edges of the parent.
    auto sub = img | box2d(vint2(3, 2), vint2(6, 4));
    export_stride("slice_3.2_6.4", sub);
    export_hash("slice_3.2_6.4", sub);
    assert(sub.nrows() == 4 && sub.ncols() == 3);
    assert(&sub(0, 0) == &img(3, 2));

    // Write the last column of the slice: must land inside the parent
    // allocation and be visible through the parent.
    for (int r = 0; r < sub.nrows(); r++)
      sub(r, sub.ncols() - 1) = 1000 + r;
    for (int r = 0; r < sub.nrows(); r++)
      assert(img(3 + r, 4) == 1000 + r);

    // Clone of a slice == slice of the values.
    auto sub_clone = clone(sub);
    export_hash("slice_clone", sub_clone);
    assert(pixel_hash(sub_clone) == pixel_hash(sub));

    // Slice with border access: the parent border covers the slice's
    // border ring, so every access stays inside the allocation.
    auto edge = img | box2d(vint2(0, 0), vint2(6, 4)); // full domain
    assert(edge.border() == img.border());
    for (int r = -2; r < 9; r++)
      for (int c = -2; c < 7; c++)
        edge(r, c) = edge(r, c) + 1; // ASAN: bounded by the parent border.
    export_hash("slice_full_after_border_write", edge);
  }

  // ---------------------------------------------------------------
  // 2. External shared buffer, positive pitch with row padding.
  // ---------------------------------------------------------------
  {
    const int R = 5, C = 7; // odd width
    const int border = 1;
    const int stride = C + 2; // elements per row, column border included
    const int pitch = stride * sizeof(int);
    // Rows span [-1, R], columns span [-1, C].
    std::vector<int> buf((R + 2) * stride, 0);
    int* begin = buf.data() + stride + 1; // row 0, col 0

    image2d<int> img(make_box2d(R, C), _data = begin, _pitch = pitch, _border = border);
    export_stride("ext_pos_padded", img);

#ifndef VPP_HEAD_CROSSCHECK
    // The addressable range covers exactly domain_with_border().
    assert((char*) img.data() == (char*) buf.data());
    assert((char*) img.data_end() == (char*) buf.data() + (R + 2) * pitch);
#endif
    assert(img.has(begin));
    assert(img.has((int*) ((char*) img.data_end() - 1)));
    assert(!img.has((int*) ((char*) img.data() - 1)));
    assert(!img.has((int*) img.data_end()));

    fill_pattern(img);
    export_hash("ext_pos_padded", img);

    // Slice the shared buffer and write its last column; the parent
    // buffer must see the writes at the right addresses.
    auto sub = img | box2d(vint2(1, 3), vint2(4, 6));
    export_stride("ext_pos_slice", sub);
    for (int r = 0; r < sub.nrows(); r++)
      sub(r, sub.ncols() - 1) = 2000 + r;
    for (int r = 0; r < sub.nrows(); r++)
      assert(buf[(1 + r + 1) * stride + 1 + 6] == 2000 + r);
    export_hash("ext_pos_slice", sub);

    // Clone the slice of the shared buffer.
    auto cl = clone(sub);
    export_hash("ext_pos_slice_clone", cl);
    assert(pixel_hash(cl) == pixel_hash(sub));
  }

#ifndef VPP_HEAD_CROSSCHECK
  // ---------------------------------------------------------------
  // 3. External shared buffer, negative pitch (bottom-up rows).
  // ---------------------------------------------------------------
  {
    const int R = 6, C = 5; // odd width
    const int border = 1;
    const int stride = C + 2; // elements per row, column borders included
    const int pitch = -int(stride * sizeof(int));
    // Address of (r, c) = begin + r * pitch + c * sizeof(int),
    // r in [-1, R], c in [-1, C]: the addressable range spans
    // [begin - R * stride - 1, begin + stride + C] in elements.
    std::vector<int> buf(R * stride + 1 + stride + C + 1, 0);
    int* begin = buf.data() + R * stride + 1;

    image2d<int> img(make_box2d(R, C), _data = begin, _pitch = pitch, _border = border);
    export_stride("ext_neg_padded", img);

    // The addressable range is bounded on both sides even though the
    // rows grow towards lower addresses.
    assert((char*) img.data() == (char*) buf.data());
    assert((char*) img.data_end() == (char*) (buf.data() + buf.size()));
    assert(img.has(begin));
    assert(img.has(&img(R - 1, C - 1)));
    assert(img.has(&img(-1, -1)));
    assert(img.has(&img(R, C)));
    assert(!img.has((int*) img.data_end()));
    assert(!img.has((int*) ((char*) img.data() - 1)));

    fill_pattern(img);
    export_hash("ext_neg_padded", img);
    for (int r = 0; r < R; r++)
      for (int c = 0; c < C; c++)
        assert(img(r, c) == pattern(r, c));

    // Same logical matrix wrapped top-down: the negative-pitch view
    // must expose identical values.
    std::vector<int> ref_buf(R * C, 0);
    image2d<int> ref(make_box2d(R, C), _data = ref_buf.data(),
                     _pitch = int(C * sizeof(int)));
    fill_pattern(ref);
    export_hash("ext_pos_tight_ref", ref);
    assert(pixel_hash(img) == pixel_hash(ref));

    // Border ring of a negative-pitch image stays inside the buffer.
    fill_border_with_value(img, 9);
    for (int c = -1; c <= C; c++)
    {
      assert(img(-1, c) == 9);
      assert(img(R, c) == 9);
    }

    // Clone: the clone is a regular top-down image with the same
    // logical content, so the hashes must match.
    auto cl = clone(img);
    export_stride("ext_neg_clone", cl);
    export_hash("ext_neg_clone", cl);
    assert(pixel_hash(cl) == pixel_hash(ref));
    for (auto p : img.domain())
      assert(cl(p) == img(p));

    // Slice a negative-pitch image and write its last column.
    auto sub = img | box2d(vint2(0, 2), vint2(5, 4));
    export_stride("ext_neg_slice", sub);
    for (int r = 0; r < sub.nrows(); r++)
      sub(r, sub.ncols() - 1) = 3000 + r;
    for (int r = 0; r < sub.nrows(); r++)
      assert(img(r, 4) == 3000 + r);
    export_hash("ext_neg_slice", sub);
    assert(pixel_hash(clone(sub)) == pixel_hash(sub));
  }
#endif

  // ---------------------------------------------------------------
  // 4. Convert: colorspace conversion of owned and sliced images.
  // ---------------------------------------------------------------
  {
    image2d<vuchar3> rgb(9, 7, _border = 1);
    pixel_wise(rgb.domain(), rgb) | [] (vint2 p, vuchar3& v) {
      v = vuchar3{(unsigned char) ((p[0] * 3 + p[1]) % 256),
                  (unsigned char) ((p[0] + p[1] * 5) % 256),
                  (unsigned char) ((p[0] * 7 + p[1] * 11) % 256)};
    };

    auto gray = rgb_to_graylevel<unsigned char>(rgb);
    export_stride("rgb_9x7", rgb);
    export_hash("rgb_9x7", rgb);
    export_hash("gray_9x7", gray);
    for (auto p : rgb.domain())
      assert(gray(p) == (rgb(p)[0] + rgb(p)[1] + rgb(p)[2]) / 3);

    // Convert a slice: values must match the full-image conversion.
    auto sub_rgb = rgb | box2d(vint2(2, 1), vint2(8, 5));
    auto sub_gray = rgb_to_graylevel<unsigned char>(sub_rgb);
    export_hash("gray_slice", sub_gray);
    for (auto p : sub_gray.domain())
      assert(sub_gray(p) == gray(p[0] + 2, p[1] + 1));

    // Round trip back to rgb.
    auto back = graylevel_to_rgb<vuchar3>(sub_gray);
    export_hash("gray_slice_rgb", back);
    for (auto p : back.domain())
      assert(back(p)[0] == sub_gray(p) && back(p)[2] == sub_gray(p));
  }

  // ---------------------------------------------------------------
  // 5. 3D odd-size image: slice, rows array and copy.
  // ---------------------------------------------------------------
  {
    imageNd<int, 3> img3(3, 5, 7);
    for (int s = 0; s < 3; s++)
      for (int r = 0; r < 5; r++)
        for (int c = 0; c < 7; c++)
          img3(s, r, c) = pattern(r + s * 5, c);

    export_stride("owned_3x5x7", img3);
    export_hash("owned_3x5x7", img3);

    auto sub = img3 | box3d(vint3(1, 1, 2), vint3(2, 4, 6));
    export_stride("slice3d", sub);
    export_hash("slice3d", sub);
    assert(&sub(0, 0, 0) == &img3(1, 1, 2));
#ifndef VPP_HEAD_CROSSCHECK
    // The row array of the slice is aligned with its first pixel.
    assert(sub[0] == &sub(0, 0, 0));
#endif

    // Write the last column of the last row of the last slice.
    sub(1, 3, 4) = 4242;
    assert(img3(2, 4, 6) == 4242);

    // Manual copy (pixel_wise is 2D-only): same logical content.
    imageNd<int, 3> cp(sub.domain(), _border = sub.border());
    for (auto p : sub.domain())
      cp(p) = sub(p);
    export_hash("slice3d_copy", cp);
    assert(pixel_hash(cp) == pixel_hash(sub));
  }

#ifndef VPP_HEAD_CROSSCHECK
  // ---------------------------------------------------------------
  // 6. 3D odd-size image with border: the slice stride must match
  //    the allocation layout so border access stays in bounds.
  // ---------------------------------------------------------------
  {
    imageNd<int, 3> img3(3, 5, 7, _border = 1);
    for (int s = 0; s < 3; s++)
      for (int r = 0; r < 5; r++)
        for (int c = 0; c < 7; c++)
          img3(s, r, c) = pattern(r + s * 5, c);

    export_stride("owned_3x5x7_b1", img3);
    export_hash("owned_3x5x7_b1", img3);

    // Every pixel of the border ring is inside the allocation.
    for (auto p : img3.domain_with_border())
      img3(p) = img3(p) + 1;

    // Restore the domain values after the border write.
    for (int s = 0; s < 3; s++)
      for (int r = 0; r < 5; r++)
        for (int c = 0; c < 7; c++)
          img3(s, r, c) = pattern(r + s * 5, c);

    auto sub = img3 | box3d(vint3(1, 1, 2), vint3(2, 4, 6));
    export_stride("slice3d_b1", sub);
    export_hash("slice3d_b1", sub);
    assert(&sub(0, 0, 0) == &img3(1, 1, 2));
    assert(sub[0] == &sub(0, 0, 0));

    sub(1, 3, 4) = 4242;
    assert(img3(2, 4, 6) == 4242);

    imageNd<int, 3> cp(sub.domain(), _border = sub.border());
    for (auto p : sub.domain())
      cp(p) = sub(p);
    export_hash("slice3d_b1_copy", cp);
    assert(pixel_hash(cp) == pixel_hash(sub));
  }
#endif

  std::cout << "OK" << std::endl;
  return 0;
}
