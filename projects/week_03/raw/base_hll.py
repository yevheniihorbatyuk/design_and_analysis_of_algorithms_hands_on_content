import mmh3  # Для хешування
import math

class HyperLogLog:
    """
    Реалізація HyperLogLog для підрахунку унікальних елементів.
    """
    def __init__(self, precision: int):
        self.precision = precision
        self.m = 1 << precision
        self.registers = [0] * self.m
        self.alpha = self._get_alpha()
    
    def _get_alpha(self) -> float:
        if self.m == 16:
            return 0.673
        elif self.m == 32:
            return 0.697
        elif self.m == 64:
            return 0.709
        return 0.7213 / (1 + 1.079 / self.m)
    
    def add(self, item: str) -> None:
        """Додає елемент до HyperLogLog."""
        x = mmh3.hash(str(item), signed=False)
        j = x & (self.m - 1)
        w = x >> self.precision
        self.registers[j] = max(self.registers[j], self._get_rho(w))
    
    def _get_rho(self, w: int) -> int:
        """Повертає позицію першої 1 в бінарному представленні."""
        return len(bin(w | 1)) - len(bin(w).rstrip('0'))
    
    def estimate(self) -> float:
        """Оцінює кількість унікальних елементів."""
        sum_inv = sum(math.pow(2.0, -x) for x in self.registers)
        estimate = self.alpha * float(self.m * self.m) / sum_inv
        
        # Корекція для малих значень
        if estimate <= 2.5 * self.m:
            zeros = self.registers.count(0)
            if zeros != 0:
                estimate = self.m * math.log(self.m / zeros)
        
        # Корекція для великих значень
        if estimate > pow(2, 32) / 30:
            estimate = -pow(2, 32) * math.log(1 - estimate / pow(2, 32))
            
        return estimate



if __name__ == "__main__":
    # Тестування
    hll = HyperLogLog(10)  # Використовуємо 2^10 регістрів
    elements = [f"user_{i}" for i in range(1, 10000)]

    for elem in elements:
        hll.add(elem)

    print("Оцінка кількості унікальних елементів:", round(hll.estimate()))