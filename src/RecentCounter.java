import java.util.ArrayDeque;
import java.util.Deque;

/**
 * @Author: Wang Xinxiang
 * @Description: 933. 最近的请求次数
 * @DateTime: 7/30/2024 10:48 AM
 */

public class RecentCounter {
    Deque<Integer> stack;
    public RecentCounter() {
        this.stack = new ArrayDeque<>();
    }

    public int ping(int t) {
        stack.addLast(t);
        int start = t-3000;
        while(stack.peekFirst()<start){
            stack.pollFirst();
        }
        return stack.size();
    }
}
