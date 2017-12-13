Range.o: Range.cpp Range.hpp
Box.o: Box.cpp common.hpp otype.hpp \
 Box.hpp Range.hpp
Partition.o: Partition.cpp Partition.hpp common.hpp otype.hpp \
 Box.hpp Range.hpp PartitionPool.hpp
Array.o: Array.cpp Array.hpp Partition.hpp common.hpp otype.hpp \
 Box.hpp Range.hpp utils/utils.hpp \
 utils/../common.hpp ArrayPool.hpp PartitionPool.hpp
Internal.o: Internal.cpp Internal.hpp common.hpp otype.hpp \
 Box.hpp Range.hpp utils/utils.hpp \
 utils/../common.hpp
Function.o: Function.cpp Function.hpp common.hpp otype.hpp \
 utils/../common.hpp Internal.hpp Box.hpp Range.hpp ArrayPool.hpp \
 Array.hpp Partition.hpp PartitionPool.hpp
Kernel.o: Kernel.cpp Kernel.hpp NodePool.hpp common.hpp otype.hpp \
 Node.hpp Array.hpp Partition.hpp Box.hpp Range.hpp ArrayPool.hpp \
 PartitionPool.hpp Function.hpp utils/utils.hpp \
 utils/../common.hpp Internal.hpp NodeDesc.hpp
Operator.o: Operator.cpp Operator.hpp NodePool.hpp common.hpp otype.hpp \
 Node.hpp Array.hpp Partition.hpp Box.hpp Range.hpp ArrayPool.hpp \
 PartitionPool.hpp Function.hpp utils/utils.hpp \
 utils/../common.hpp Internal.hpp NodeDesc.hpp Kernel.hpp Jit_Driver.hpp \
 jit/Jit.hpp
Node.o: Node.cpp Node.hpp Array.hpp Partition.hpp common.hpp otype.hpp \
 Box.hpp Range.hpp NodeDesc.hpp Operator.hpp NodePool.hpp ArrayPool.hpp \
 PartitionPool.hpp Function.hpp utils/utils.hpp \
 utils/../common.hpp Internal.hpp
IO.o: IO.cpp IO.hpp Array.hpp Partition.hpp common.hpp otype.hpp \
 Box.hpp Range.hpp utils/utils.hpp \
 Function.hpp Internal.hpp ArrayPool.hpp PartitionPool.hpp
utils.o: utils/utils.cpp utils/utils.hpp \
 utils/../common.hpp utils/../otype.hpp 
c_oa_type.o: c-interface/c_oa_type.cpp c-interface/c_oa_type.hpp \
 c-interface/../ArrayPool.hpp c-interface/../common.hpp \
 c-interface/../otype.hpp \
 c-interface/../Array.hpp c-interface/../Partition.hpp \
 c-interface/../Box.hpp c-interface/../Range.hpp \
 c-interface/../PartitionPool.hpp c-interface/../NodePool.hpp \
 c-interface/../Node.hpp c-interface/../ArrayPool.hpp \
 c-interface/../Function.hpp c-interface/../utils/utils.hpp \
 c-interface/../utils/../common.hpp c-interface/../Internal.hpp \
 c-interface/../Function.hpp c-interface/../Operator.hpp \
 c-interface/../NodePool.hpp c-interface/../NodeDesc.hpp
c_oa_utils.o: c-interface/c_oa_utils.cpp c-interface/c_oa_utils.hpp
main.o: main.cpp test/test.hpp test/../Range.hpp test/../Box.hpp \
 test/../common.hpp test/../otype.hpp \
 test/../Range.hpp test/../Partition.hpp test/../Box.hpp \
 test/../Array.hpp test/../Partition.hpp test/../Function.hpp \
 test/../utils/utils.hpp \
 test/../utils/../common.hpp test/../Internal.hpp test/../ArrayPool.hpp \
 test/../Array.hpp test/../PartitionPool.hpp test/../Internal.hpp \
 test/../Operator.hpp test/../NodePool.hpp test/../Node.hpp \
 test/../Function.hpp test/../NodeDesc.hpp test/../IO.hpp \
 test/../c-interface/c_oa_type.hpp test/../c-interface/../ArrayPool.hpp \
 test/../c-interface/../NodePool.hpp
 Grid.o:Grid.cpp

oa_partition.o \
 oa_partition.mod: \
 fortran/oa_partition.F90

oa_type.o oa_type.mod: \
 fortran/oa_type.F90 fortran/config.h \

oa_ops.o oa_ops.mod: \
 fortran/oa_ops.F90 fortran/config.h oa_type.mod

oa_utils.o oa_utils.mod: \
 fortran/oa_utils.F90

oa_sub.o oa_sub.mod: \
 fortran/oa_sub.F90 oa_type.mod

oa_mod.o oa_mod.mod: \
 fortran/oa_mod.F90 \
 oa_type.mod oa_utils.mod oa_ops.mod oa_partition.mod oa_sub.mod

oa_test.o oa_test.mod: \
 fortran/oa_test.F90 oa_mod.mod

oa_main.o test.mod: \
 fortran/oa_main.F90 oa_mod.mod oa_test.mod \
 test.mod
 