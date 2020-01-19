# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 01:37:40 2019

@author: youjibiying
"""

from typing import List
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        summation = 0
        L, R = 0, -1
        optim = len(nums) + 1
        while R < len(nums):
            while R < len(nums):
                R += 1
                if R < len(nums):
                    summation += nums[R]
                if summation >= s:
                    optim = min(optim, R - L + 1)
                    break
    
            if R == len(nums):
                break
    
            while L < R:
                summation -= nums[L]
                L += 1
                if summation >= s:
                    optim = min(optim, R - L + 1)
                else:
                    break
        return optim if optim != len(nums) + 1 else 0

if __name__=='__main__':
    so=Solution()
    nums=[2,3,1,2,4,3]

    print(so.minSubArrayLen(7,nums))