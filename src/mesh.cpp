#include <maskgen/mesh.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <BRep_Tool.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <Bnd_Box.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <Poly_Array1OfTriangle.hxx>
#include <Poly_Triangulation.hxx>
#include <STEPControl_Reader.hxx>
#include <TColgp_Array1OfPnt.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <TopExp_Explorer.hxx>
#include <TopLoc_Location.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Vertex.hxx>
#include <gp_Pnt.hxx>

namespace maskgen {

Mesh::Mesh() = default;

Mesh::~Mesh() = default;

const std::vector<float>& Mesh::vertices() const { return vertices_; }

const std::vector<uint32_t>& Mesh::indices() const { return indices_; }

bool Mesh::empty() const { return vertices_.empty() || indices_.empty(); }

bool Mesh::LoadFromFile(const std::string& path, float scale) {
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".ply") {
    return LoadFromPly(path);
  }
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".obj") {
    return LoadFromObj(path);
  }
  if (path.size() >= 5 && path.substr(path.size() - 5) == ".step") {
    return LoadFromStep(path, scale);
  }
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".stp") {
    return LoadFromStep(path, scale);
  }
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".stl") {
    return LoadFromStl(path);
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

bool Mesh::LoadFromStep(const std::string& path, float scale) {
  STEPControl_Reader reader;
  IFSelect_ReturnStatus status = reader.ReadFile(path.c_str());
  if (status != IFSelect_RetDone) {
    return false;
  }

  if (reader.TransferRoots() == 0) {
    return false;
  }

  TopoDS_Shape shape = reader.OneShape();
  if (shape.IsNull()) {
    return false;
  }

  Bnd_Box bbox;
  for (TopExp_Explorer exp(shape, TopAbs_VERTEX); exp.More(); exp.Next()) {
    bbox.Add(BRep_Tool::Pnt(TopoDS::Vertex(exp.Current())));
  }
  double deflection = 0.1;
  if (!bbox.IsVoid()) {
    Standard_Real xmin, ymin, zmin, xmax, ymax, zmax;
    bbox.Get(xmin, ymin, zmin, xmax, ymax, zmax);
    double diag = std::sqrt((xmax - xmin) * (xmax - xmin) +
                            (ymax - ymin) * (ymax - ymin) +
                            (zmax - zmin) * (zmax - zmin));
    if (diag > 0) {
      deflection = diag * 0.001;
    }
  }

  BRepMesh_IncrementalMesh mesher(shape, deflection);
  mesher.Perform();
  if (!mesher.IsDone()) {
    return false;
  }

  vertices_.clear();
  indices_.clear();

  uint32_t vertex_offset = 0;

  for (TopExp_Explorer exp(shape, TopAbs_FACE); exp.More(); exp.Next()) {
    const TopoDS_Face& face = TopoDS::Face(exp.Current());
    TopLoc_Location loc;
    Handle(Poly_Triangulation) tri = BRep_Tool::Triangulation(face, loc);
    if (tri.IsNull()) {
      continue;
    }

    const gp_Trsf& trsf = loc.Transformation();

    int nb_nodes = tri->NbNodes();
    vertices_.reserve(vertices_.size() + nb_nodes * 3);

    for (int i = 1; i <= nb_nodes; ++i) {
      gp_Pnt p = tri->Node(i).Transformed(trsf);
      vertices_.push_back(static_cast<float>(p.X() * scale));
      vertices_.push_back(static_cast<float>(p.Y() * scale));
      vertices_.push_back(static_cast<float>(p.Z() * scale));
    }

    bool forward = face.Orientation() == TopAbs_FORWARD;
    int nb_triangles = tri->NbTriangles();
    indices_.reserve(indices_.size() + nb_triangles * 3);

    for (int i = 1; i <= nb_triangles; ++i) {
      Standard_Integer n1, n2, n3;
      tri->Triangle(i).Get(n1, n2, n3);
      n1 -= 1;
      n2 -= 1;
      n3 -= 1;
      if (forward) {
        indices_.push_back(vertex_offset + n1);
        indices_.push_back(vertex_offset + n2);
        indices_.push_back(vertex_offset + n3);
      } else {
        indices_.push_back(vertex_offset + n1);
        indices_.push_back(vertex_offset + n3);
        indices_.push_back(vertex_offset + n2);
      }
    }

    vertex_offset += static_cast<uint32_t>(nb_nodes);
  }

  return !empty();
}

bool Mesh::LoadFromStl(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  file.seekg(0, std::ios::end);
  auto file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  char header[80];
  if (!file.read(header, 80)) {
    return false;
  }

  uint32_t num_triangles = 0;
  if (!file.read(reinterpret_cast<char*>(&num_triangles), 4)) {
    return false;
  }

  auto expected_size =
      std::streamoff(84) + static_cast<std::streamoff>(num_triangles) * 50;
  if (file_size == expected_size) {
    vertices_.clear();
    vertices_.reserve(num_triangles * 9);
    indices_.clear();
    indices_.reserve(num_triangles * 3);

    for (uint32_t i = 0; i < num_triangles; ++i) {
      float normal[3];
      float v[3][3];
      uint16_t attr;
      file.read(reinterpret_cast<char*>(normal), 12);
      file.read(reinterpret_cast<char*>(v), 36);
      file.read(reinterpret_cast<char*>(&attr), 2);

      uint32_t base = static_cast<uint32_t>(vertices_.size() / 3);
      for (int j = 0; j < 3; ++j) {
        vertices_.push_back(v[j][0]);
        vertices_.push_back(v[j][1]);
        vertices_.push_back(v[j][2]);
      }
      indices_.push_back(base);
      indices_.push_back(base + 1);
      indices_.push_back(base + 2);
    }

    return !empty();
  }

  file.close();
  file.open(path);
  if (!file.is_open()) {
    return false;
  }

  vertices_.clear();
  indices_.clear();

  std::string line;
  int vert_count = 0;

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string token;
    iss >> token;
    if (token != "vertex") {
      continue;
    }
    float x, y, z;
    iss >> x >> y >> z;
    vertices_.push_back(x);
    vertices_.push_back(y);
    vertices_.push_back(z);
    ++vert_count;
    if (vert_count % 3 == 0) {
      uint32_t base = static_cast<uint32_t>(vertices_.size() / 3) - 3;
      indices_.push_back(base);
      indices_.push_back(base + 1);
      indices_.push_back(base + 2);
    }
  }

  return !empty();
}

}  // namespace maskgen
