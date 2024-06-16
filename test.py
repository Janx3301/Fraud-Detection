class Solution(object):
    def threeSum(self, nums):
        n = len(nums)
        ans = []
        nums.sort()
        for i in range(n):
            if i!=0 and nums[i]==nums[i-1]:
                continue
            j = i+1
            k = n-1
            while j<k:
                total = nums[i]+nums[j]+nums[k]
                if total>0:
                    k-=1
                elif total<0:
                    j+=1
                else:
                    temp = [nums[i],nums[j],nums[k]]
                    ans.append(temp)
                    j+=1
                    k-=1
                    while j<k and nums[j]==nums[j-1]:
                        j+=1
                    while j<k and nums[k]==nums[k+1]:
                        k-=1
        return ans

nums = [-1, 0, 1, 2, -1, -4]
solution = Solution()
print(*solution.threeSum(nums))
       
        