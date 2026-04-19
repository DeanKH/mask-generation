#include <maskgen/mesh.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace maskgen {

Mesh::Mesh() = default;

Mesh::~Mesh() = default;

const std::vector<float>& Mesh::vertices() const { return vertices_; }

const std::vector<uint32_t>& Mesh::indices() const { return indices_; }

bool Mesh::empty() const { return vertices_.empty() || indices_.empty(); }

bool Mesh::LoadFromFile(const std::string& path) {
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".ply") {
    return LoadFromPly(path);
  }
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".obj") {
    return LoadFromObj(path);
  }
  return false;
}

bool Mesh::LoadFromPly(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }

  std::string line;
  int vertex_count = 0;
  int face_count = 0;
  bool in_header = true;

  while (in_header && std::getline(file, line)) {
    std::istringstream iss(line);
    std::string token;
    iss >> token;

    if (token == "element") {
      std::string type;
      int count;
      iss >> type >> count;
      if (type == "vertex") {
        vertex_count = count;
      } else if (type == "face") {
        face_count = count;
      }
    } else if (token == "end_header") {
      in_header = false;
    }
  }

  vertices_.clear();
  vertices_.reserve(vertex_count * 3);
  indices_.clear();
  indices_.reserve(face_count * 3);

  for (int i = 0; i < vertex_count; ++i) {
    if (!std::getline(file, line)) {
      return false;
    }
    std::istringstream iss(line);
    float x, y, z;
    iss >> x >> y >> z;
    vertices_.push_back(x);
    vertices_.push_back(y);
    vertices_.push_back(z);
  }

  for (int i = 0; i < face_count; ++i) {
    if (!std::getline(file, line)) {
      return false;
    }
    std::istringstream iss(line);
    int n_verts;
    iss >> n_verts;
    if (n_verts < 3) {
      continue;
    }
    std::vector<uint32_t> face_indices(n_verts);
    for (int j = 0; j < n_verts; ++j) {
      iss >> face_indices[j];
    }
    for (int j = 1; j < n_verts - 1; ++j) {
      indices_.push_back(face_indices[0]);
      indices_.push_back(face_indices[j]);
      indices_.push_back(face_indices[j + 1]);
    }
  }

  return !empty();
}

bool Mesh::LoadFromObj(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }

  vertices_.clear();
  indices_.clear();

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string prefix;
    iss >> prefix;

    if (prefix == "v") {
      float x, y, z;
      iss >> x >> y >> z;
      vertices_.push_back(x);
      vertices_.push_back(y);
      vertices_.push_back(z);
    } else if (prefix == "f") {
      std::vector<uint32_t> face_indices;
      std::string vertex_str;
      while (iss >> vertex_str) {
        size_t slash_pos = vertex_str.find('/');
        int idx = std::stoi(vertex_str.substr(0, slash_pos));
        if (idx < 0) {
          idx = static_cast<int>(vertices_.size() / 3) + idx + 1;
        }
        face_indices.push_back(static_cast<uint32_t>(idx - 1));
      }
      for (size_t j = 1; j + 1 < face_indices.size(); ++j) {
        indices_.push_back(face_indices[0]);
        indices_.push_back(face_indices[j]);
        indices_.push_back(face_indices[j + 1]);
      }
    }
  }

  return !empty();
}

}  // namespace maskgen
