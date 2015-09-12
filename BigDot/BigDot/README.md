-----------------------------------------------------------------------------------------------
--------------------------          Big Dot          -----------------------------------------
-----------------------------------------------------------------------------------------------

-- TeamName             : ORZ

--  Name                :  Zhizhong Pan
                        :  Wenqian Tao

--  E-mail              :  zhizhop@g.clemson.edu
                        :  wtao@g.clemson.edu

------------------------------------------------------------------------------------------------

--  Project No          :  Lab One

--  Project due date    :  09/12/14 11:59PM

--  Project description::  This project aim to use OpenCL caculate bigdot product.

-------------------------------------------------------------------------------------------------

-- Run                  :  On SoC Ubuntu machine use this command to complile:
                           g++ -I /usr/local/cuda-5.5/targets/x86_64-linux/include bigdot.cpp -o bigdot

      
-- Instruction         :  After compile you can read two file like this: 
                          ./bigdot file file
                          And it will retrun calculation result on console.

                          You also can ues validation fuciton and writeFile to test. 
                          I comment them before hand in.

-- problem?            : If use large vector and big value will cause LOSE Accuracy. The reason is the 
                         caculation betweent float. Float type will lose this accuracy deal with big value. 

----------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------

