# https://github.com/b001io/wagner-fischer.git
class spell_check:
    def __init__(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            self.dictionary = [i.strip() for i in f]

    def check(self, word):
        suggestions = []

        for correct_word in self.dictionary:
            distance = self.__wagner_fischer__(word, correct_word)
            suggestions.append((correct_word, distance))

        suggestions.sort(key=lambda x: x[1])
        return suggestions[:10]

    def __wagner_fischer__(self, s1, s2):
        len_s1, len_s2 = len(s1), len(s2)
        if len_s1 > len_s2:
            s1, s2 = s2, s1
            len_s1, len_s2 = len_s2, len_s1

        current_row = range(len_s1 + 1)
        for i in range(1, len_s2 + 1):
            previous_row, current_row = current_row, [i] + [0] * len_s1
            for j in range(1, len_s1 + 1):
                add, delete, change = previous_row[j] + 1, current_row[j-1] + 1, previous_row[j-1]
                if s1[j-1] != s2[i-1]:
                    change += 1
                current_row[j] = min(add, delete, change)

        return current_row[len_s1]

if __name__ == "__main__":
    import time

    misspelled_word = "wrlod"
    checker = spell_check("data\\words.txt")

    # start timer
    start_time = time.perf_counter()
    suggestions = checker.check(misspelled_word)
    print(f"Time taken: {(time.perf_counter() - start_time):.0f} sec")

    print(f"Top 10 suggestions for '{misspelled_word}':")
    for word, distance in suggestions:
        print(f"{word} (Distance: {distance})")
