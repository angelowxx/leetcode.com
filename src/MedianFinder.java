import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Stack;

/**
 * @Author: Wang Xinxiang
 * @Description: 295. 数据流的中位数
 * @DateTime: 7/31/2024 6:18 PM
 */

class MedianFinder {

    PriorityQueue<Integer> left;
    PriorityQueue<Integer> right;
    static double med;
    public MedianFinder() {
        left = new PriorityQueue<>((a,b)->b-a);
        right = new PriorityQueue<>();
    }

    public void addNum(int num) {
        if(left.isEmpty()&&right.isEmpty()){
            left.add(num);
            med = num;
            return;
        }
        if(num>=med){
            right.add(num);
        }
        else{
            left.add(num);
        }
        if(left.size()<right.size()){
            left.add(right.poll());
        }
        else if(left.size()>right.size()+1){
            right.add(left.poll());
        }
        if(left.size()==right.size()){
            med = left.peek()+right.peek();
            med = med/2;
        }
        else{
            med = left.peek();
        }
    }

    public double findMedian() {

        return med;
    }
}
