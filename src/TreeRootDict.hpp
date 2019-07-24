/*
 * TreeRootDict.hpp
 * Cache the builded tree
 *
=======================================================*/

#ifndef __TREEROOTDICT_HPP__
#define __TREEROOTDICT_HPP__

#include "Array.hpp"
#include "common.hpp"
#include "PartitionPool.hpp"
#include <list>
#include <vector>
#include <stack>
#include <unordered_map>
#include "modules/tree_tool/NodeVec.hpp"
#include "modules/tree_tool/Simple_Node.hpp"

using namespace oa::ops;

// each hash value mapping to an arraylist
typedef std::unordered_map<size_t, NodePtr> TreeRootMap;

class TreeRootDict
{
  private:
    TreeRootMap m_map; // use [Partition info, buffer_data_type] as identical key

  public:
    TreeRootDict() {}

    ~TreeRootDict(){
	m_map.clear(); 
    }
 
    void clear(){
	m_map.clear();
    }

    // show TreeRootDict's status, for debug
    void show_status()
    {
        printf(" == == == == == Tree Root Dictory STATUS == == == == == == == =\n ");
        printf("Tree Root Size: %d\n", m_map.size());

        return;
    }

    void insert(size_t hash, NodePtr np)
    {
        m_map[hash] = np;
        return;
    }


    bool find(size_t hash){
        auto iter = m_map.find(hash);
        return iter != m_map.end();
    }

    int size(){
        return m_map.size();
    }
    

    NodePtr& get(size_t hash)
    {
        auto iter = m_map.find(hash);
        return iter->second;
    }

    static TreeRootDict *global()
    {
        static TreeRootDict td;
        return &td;
    }
};

typedef std::unordered_map <size_t, vector<NodePtr> > TreeDataNodesMap;
class TreeDataNodes
{
  private:
    TreeDataNodesMap m_map;

  public:
   TreeDataNodes(){}
   ~TreeDataNodes(){
	m_map.clear();
}
   
    void insert(size_t hash, vector<NodePtr> &np_vector)
    {
        m_map[hash] = vector<NodePtr>(); 
        auto& it = m_map[hash];
        it.assign(np_vector.begin(), np_vector.end());
        return;
    }

    bool find(size_t hash){
        auto iter = m_map.find(hash);
        return iter != m_map.end();
    }

    vector<NodePtr>& get(size_t hash)
    {
        auto it = m_map.find(hash);
        return it->second;
    }

    void modify(size_t hash, vector<Simple_node>& new_data_nodes_vector)
    {
        auto iter = m_map.find(hash);
        auto& data_nodes_vector = iter->second;
	    int simp_datasize = NodeVec::global()->get_datanode_size();

        for (int i = 0; i < simp_datasize; ++i)
        {
            auto& cur_node = data_nodes_vector[i];
            auto& cur_simp_node = new_data_nodes_vector[i];
            int node_type = cur_simp_node.type;
            if(node_type == TYPE_DATA){
                 cur_node->set_data(ArrayPtr(cur_simp_node.get_ArrayPtr()));
            }

            else if(node_type == TYPE_INT3_REP || node_type == TYPE_INT3_SHIFT) {
                   int *h_val_ptr = (int*)cur_simp_node.get_val();
                   ArrayPtr& ap = cur_node->get_data();
                   #ifndef __HAVE_CUDA__
                   oa::internal::copy_buffer((int*)ap->get_buffer(), (int*)h_val_ptr, 3);
                   #else
                   int *d_val_ptr;
                   CUDA_CHECK(cudaMalloc((void**)&d_val_ptr, sizeof(int)*3));
                   CUDA_CHECK(cudaMemcpy(d_val_ptr, h_val_ptr, sizeof(int)*3, cudaMemcpyHostToDevice));
                   oa::gpu::copy_buffer((int*)ap->get_buffer(), (int*)d_val_ptr, 3);
                   CUDA_CHECK(cudaFree(d_val_ptr));
                   #endif
            }

            else if(node_type == TYPE_REP || node_type == TYPE_REF){
                    int *val_ptr = (int*)cur_simp_node.get_val();
                    const Box b = Box(val_ptr[0],val_ptr[1],val_ptr[2],val_ptr[3],val_ptr[4],val_ptr[5]);
                    cur_node->set_ref(b);
            }

            else if(node_type == TYPE_INT ||node_type == TYPE_FLOAT || node_type == TYPE_DOUBLE){
                void *val_ptr = cur_simp_node.get_val();
                ArrayPtr& ap = cur_node->get_data();
                Box box = ap->get_local_box();
                int size = box.size_with_stencil(0);
                #ifndef __HAVE_CUDA__
                switch (node_type)
                {
                case TYPE_INT:
                    oa::internal::set_buffer_consts((int*)ap->get_buffer(), size, *((int*)val_ptr));
                    break;
                case TYPE_FLOAT:
                    oa::internal::set_buffer_consts((float*)ap->get_buffer(), size, *((float*)val_ptr));
                    break;
                case TYPE_DOUBLE:
                    oa::internal::set_buffer_consts((double *)ap->get_buffer(), size, *((double*)val_ptr));
                    break;
                }
                #else
                switch (node_type)
                {
                case TYPE_INT:
                    oa::gpu::set_buffer_consts((int*)ap->get_buffer(), size, *((int*)val_ptr));
                    break;
                case TYPE_FLOAT:
                    oa::gpu::set_buffer_consts((float*)ap->get_buffer(), size, *((float*)val_ptr));
                    break;
                case TYPE_DOUBLE:
                    oa::gpu::set_buffer_consts((double *)ap->get_buffer(), size, *((double*)val_ptr));
                    break;
                }
                #endif
            }
	    
            else if (node_type == TYPE_UNKNOWN)
            {
                int *val_ptr = (int*)cur_simp_node.get_val();
                cur_node->set_slice(*val_ptr);
            }

            else {
                std::cout<<"==============Get wrong node type when modify the tree, exit!==================\n";
                std::cout<<"Node Type: "<<node_type<<endl;
                std::cout<<"Tree hash: "<<hash<<endl;
                exit(EXIT_FAILURE);
            }
            
        }
        return;
    }

    static TreeDataNodes *global()
    {
        static TreeDataNodes tp;
        return &tp;
    }

    void clear(){
        m_map.clear();
    }
};


void data_vec_clear();
NodePtr& oa_build_tree();
void tree_clear();
#endif
