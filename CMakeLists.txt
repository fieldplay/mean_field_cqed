#
# @BEGIN LICENSE
#
# myplugin by Psi4 Developer, a plugin to:
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2016 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# These were the compilers used to compile Psi4. They may or may not be
# required to successfully run your plugin.
#set(CMAKE_C_COMPILER /edfs/users/edgroup/software/intel2016/bin/icc)
#set(CMAKE_CXX_COMPILER /edfs/users/edgroup/software/intel2016/bin/icpc)
#set(CMAKE_Fortran_COMPILER )
set(PYTHON_EXECUTABLE /usr/bin/python)
project(mean_field_cqed CXX)

find_package(psi4 1.0 REQUIRED)

add_psi4_plugin(mean_field_cqed frozen_natural_orbitals.cc  tdhf.cc  tdhf_cqed.cc)
