class Calculator:
    @staticmethod
    def multiply(a, b):
        """
        Multiply two numbers. Coerces numeric strings (which LLM tool-calls
        sometimes produce, e.g. "2500" instead of 2500) to floats first so a
        stray string type never causes a silent TypeError mid-conversation.

        Args:
            a: first number (int, float, or numeric string)
            b: second number (int, float, or numeric string)

        Returns:
            float: The product of a and b.
        """
        return float(a) * float(b)
    
    @staticmethod
    def calculate_total(*x) -> float:
        """
        Calculate sum of the given list of numbers

        Args:
            x (list): List of numbers (int, float, or numeric strings)

        Returns:
            float: The sum of numbers in the list x
        """
        return sum(float(v) for v in x)

    @staticmethod
    def calculate_daily_budget(total, days) -> float:
        """
        Calculate daily budget

        Args:
            total: Total cost (int, float, or numeric string).
            days: Total number of days (int, float, or numeric string).

        Returns:
            float: Expense for a single day
        """
        total, days = float(total), float(days)
        return total / days if days > 0 else 0
    
    
