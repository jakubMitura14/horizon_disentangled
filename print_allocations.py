from allocation_logic import get_final_allocations

df_alloc_y1, df_alloc_y2, df_alloc_y3 = get_final_allocations()

print("--- Year 1 Allocations ---")
print(df_alloc_y1)
print("\n--- Year 2 Allocations ---")
print(df_alloc_y2)
print("\n--- Year 3 Allocations ---")
print(df_alloc_y3)
