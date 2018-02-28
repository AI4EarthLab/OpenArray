#!/bin/bash

#build_dir=./build
build_dir=`pwd`/build
current_dir=`pwd`

osname=`uname`

if [ ! -d "$build_dir" ]; then
    mkdir ${build_dir}
fi

cp makefile*  ${build_dir}/ 2>/dev/null
sed -i "s/##===##//g" ${build_dir}/makefile*

for d in `find . -maxdepth 2 -type d`
do

    full=$current_dir/`echo $d | cut -c3-`

    if [[ $full == *"build"* ]]
    then
    continue
    fi
    
    if [ ! -d "${build_dir}/${d}" ]; then
        mkdir ${build_dir}/${d}
    fi
    
    for f in `find $full -maxdepth 1 -type f \
              \( -iname \*.F90     \
                 -o -iname \*.h    \
                 -o -iname \*.h90  \
                 -o -iname \*.hpp  \
                 -o -iname \*.cpp  \
                 -o -iname \*.c \) \
              2>/dev/null`
    do
        filename="${f%.*}"
        extension="${f##*.}"

    filename=$(basename $filename)
    #echo $filename.$extension

    src_filename=$full/$filename.$extension
    dst_filename=${build_dir}/`echo ${d} | cut -c3-`/$filename.$extension

    # echo $src_filename
    # echo $dst_filename
    
        #m4 ${filename}.${extension} > \
    #   ${build_dir}/$filename.${extension}
        #./fypp ${filename}.${extension} > \
    #  ${build_dir}/$filename.${extension} -m re
    

    #echo ./${build_dir}/$f
        #echo $f | cut -d '\.\/' -f 2
    
    if [ ! -f "$dst_filename" ]; then
      ./fypp -p -m re -m string -m io -m os --create-parents \
           $src_filename > $dst_filename &
      echo " >>>processing file $src_filename" 
    else
      if [ "$osname" = "Linux" ];then 
        src_time=$(date +'%y%m%d%H%M%S' -r $src_filename)
        dst_time=$(date +'%y%m%d%H%M%S' -r $dst_filename)
      else
        src_time=$(stat -f "%Sm" -t "%y%m%d%H%M%S" $src_filename)
        dst_time=$(stat -f "%Sm" -t "%y%m%d%H%M%S" $dst_filename)
      fi
        #src_time=$(stat -f "%Sm" -t "%y%m%d%H%M" $src_filename)
        #dst_time=$(stat -f "%Sm" -t "%y%m%d%H%M" $dst_filename)
      if [[ "$src_time" > "$dst_time" ]]; then
        ./fypp -p -m re -m string -m io -m os --create-parents \
           $src_filename > $dst_filename
        
        echo " >>>processing file $src_filename"
      fi
    fi

    done
done
wait
