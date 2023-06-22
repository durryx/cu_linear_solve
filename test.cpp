#include <iostream>

unsigned int f_n(unsigned int x, unsigned int y, int counter)
{
    if (counter == 0)
        return x + y;
    if (y == 0)
        return x;
    return f_n(f_n(x, y - 1, counter), f_n(x, y - 1, counter) + y, counter - 1);
}

int main() { std::cout << f_n(3, 1, 2) << '\n'; }