




## Big calcuation code:

```C++
#include <iostream>
#include <omp.h>

using namespace std;

int main()
{
	unsigned long long accSum = 0;

	#pragma omp parallel for reduction(+:accSum) //reduction creates local accSum for each thread. In another case you would have race condition)
	for (int i = 0; i < 100000; i++) {
		for (int j = 0; j < 100000; j++) {
			accSum += i * j;
		}
	}

	cout << accSum << endl;

	return 0;
}
```
