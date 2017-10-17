class Grid {
  private:
  char grid_type;
  vector<Grid_DistancePtr> distances;
  vector<int> x_map;
  vector<int> y_map;
  vector<int> z_map;

  public:
  void init_grid(char type);
  ArrayPtr get_grid_dx(ArrayPtr u);
  ArrayPtr get_grid_dy(ArrayPtr u);
  ArrayPtr get_grid_dz(ArrayPtr u);

};
