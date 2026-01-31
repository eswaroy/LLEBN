try:
    with open("debug_output.txt", "w") as f:
        f.write("Python File Write Success")
except Exception as e:
    print(f"Failed: {e}")
