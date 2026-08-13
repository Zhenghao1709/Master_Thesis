# main_build_all_turbines.py

from __future__ import annotations

from main_build_healthy_data import main as build_one_turbine


def main():
    turbine_list = [f"Kelmarsh_{i}" for i in range(1, 7)]

    for turbine_name in turbine_list:
        try:
            build_one_turbine(turbine_name=turbine_name)
        except Exception as e:
            print(f"[ERROR] Failed for {turbine_name}: {e}")


if __name__ == "__main__":
    main()