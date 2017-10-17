class Reference : public Box{
  private:

  public:
  Reference(range rx, range ry, range rz);
  Reference(int[3] starts, int[3] counts);
  display(string prefix);
  
  
};
