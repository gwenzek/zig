ZIGDIR=$HOME/github/zig
LLVMDIR=$HOME/github/llvm-project-14.0.5.src
OUTDIR=$HOME/local/llvm
export CXX=g++

set -ex

# cd $LLVMDIR/llvm
# mkdir -p build-release
# cd build-release
# cmake .. -DCMAKE_INSTALL_PREFIX=$OUTDIR \
# 	 -DCMAKE_PREFIX_PATH=$OUTDIR \
# 	 -DCMAKE_BUILD_TYPE=Release \
# 	 -DLLVM_ENABLE_LIBXML2=OFF \
# 	 -DLLVM_ENABLE_ASSERTIONS=ON \
# 	 -DCMAKE_CXX_FLAGS="-g2" \
# 	 -G Ninja \
# 	 -DLLVM_PARALLEL_LINK_JOBS=1
# ninja install

# Lld
# cd $LLVMDIR/lld
# mkdir -p build-release
# cd build-release
# cmake .. -DCMAKE_INSTALL_PREFIX=$OUTDIR -DCMAKE_PREFIX_PATH=$OUTDIR -DCMAKE_BUILD_TYPE=Release -G Ninja -DLLVM_PARALLEL_LINK_JOBS=1 -DCMAKE_CXX_STANDARD=17
# ninja install


# Clang
# cd $LLVMDIR/clang
# mkdir -p build-release
# cd build-release
# cmake .. -DCMAKE_INSTALL_PREFIX=$OUTDIR -DCMAKE_PREFIX_PATH=$OUTDIR -DCMAKE_BUILD_TYPE=Release -G Ninja -DLLVM_PARALLEL_LINK_JOBS=1
# ninja install

mkdir -p $ZIGDIR/build; cd $ZIGDIR/build
# TODO: remove all deprecated calls
cmake .. \
	-DCMAKE_PREFIX_PATH=$OUTDIR \
	-DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations -g2" \

make VERBOSE=1 install

cd $ZIGDIR
./build/zig test -I test test/behavior.zig 
