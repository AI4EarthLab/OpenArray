#ifndef __oa__vector__
#define __oa__vector__

template <typename Object>
class oa_vector
{
  public:
    explicit oa_vector(int initSize = 0)
      : theSize(initSize), theCapacity(initSize + SPARE_CAPACITY)
    {
      objects = new Object[theCapacity];
    }

    oa_vector(const oa_vector & rhs)
      :objects(nullptr)
    {
      operator = (rhs);
    }

    ~oa_vector()
    {
      delete [] objects;
    }

    const oa_vector& operator=( const oa_vector& rhs )
    {
      if (this != & rhs)
      {
        delete [] objects;
        theSize = rhs.theSize;
        theCapacity = rhs.theCapacity;
        objects = new Object[rhs.theCapacity];
        for (int k = 0; k < rhs.theSize; k++)
        {
          objects[k] = rhs.objects[k];
        }
      }
      return *this; 
    }

    void resize(int newSize)
    {
      if (newSize > theCapacity)
      {
        reserve(newSize * 2 + 1);
      }
      theSize = newSize;
    }
    void clear()
    {
      theSize = 0;
    }
    void reserve(int newCapacity)
    {
      if (newCapacity < theSize)
      {
        return;
      }
      Object* oldArray = objects;
      objects = new Object[newCapacity];
      for (int k = 0; k < theSize; k++)
      {
        objects[k] = oldArray[k];
      }
      theCapacity = newCapacity;
      delete [] oldArray;
    }

    Object & operator[](int index)
    {
      return objects[index];
    }
    const Object& operator[](int index) const
    {
      return objects[index];
    }

    bool empty() const
    {
      return theSize == 0;
    }

    int size() const
    {
      return theSize;
    }

    int capacity() const
    {
      return theCapacity;
    }

    void push_back(const Object& x)
    {
      if (theSize == theCapacity)
      {
        reserve(2* theCapacity + 1);
      }
      objects[theSize++] = x;
    }

    void pop_back()
    {
      theSize--;
    }

    const Object& back() const
    {
      return objects[theSize - 1];
    }

    typedef Object* iterator;
    typedef const Object* const_iterator;

    iterator begin()
    {
      return &objects[0];
    }
    const_iterator begin() const
    {
      return &objects[0];
    }

    iterator end()
    {
      return &objects[size()];
    }
    const_iterator end() const
    {
      return &objects[size()];
    }

    enum { SPARE_CAPACITY = 16 };
  private:
    int theSize;
    int theCapacity;
    Object* objects;
};




#endif 

