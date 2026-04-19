#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace maskgen {

class Mesh {
 public:
  Mesh();
  ~Mesh();

  Mesh(const Mesh&) = default;
  Mesh& operator=(const Mesh&) = default;
  Mesh(Mesh&&) noexcept = default;
  Mesh& operator=(Mesh&&) noexcept = default;

  bool LoadFromFile(const std::string& path);

  const std::vector<float>& vertices() const;
  const std::vector<uint32_t>& indices() const;
  bool empty() const;

 private:
  bool LoadFromPly(const std::string& path);
  bool LoadFromObj(const std::string& path);

  std::vector<float> vertices_;
  std::vector<uint32_t> indices_;
};

}  // namespace maskgen
