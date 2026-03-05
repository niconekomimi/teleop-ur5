#!/usr/bin/env python3
import argparse
import time
import inspect
from pymodbus.client import ModbusSerialClient

def write_regs(client: ModbusSerialClient, address: int, values: list[int], slave_id: int):
    """兼容多版本 pymodbus 的 API"""
    sig = inspect.signature(client.write_registers)
    params = sig.parameters
    if "slave" in params:
        return client.write_registers(address, values, slave=slave_id)
    if "unit" in params:
        return client.write_registers(address, values, unit=slave_id)
    if "device_id" in params:
        return client.write_registers(address, values, device_id=slave_id)
    return client.write_registers(address, values, slave_id)

def activate_gripper(client: ModbusSerialClient, slave_id: int) -> None:
    print("[Robotiq] 清除故障并重启夹爪...")
    write_regs(client, 0x03E8, [0x0000, 0x0000, 0x0000], slave_id)
    time.sleep(0.5)

    print("[Robotiq] 激活夹爪 (约需 3 秒)...")
    write_regs(client, 0x03E8, [0x0900, 0x0000, 0x0000], slave_id)
    time.sleep(3.0)
    print("[Robotiq] 激活完成！")

def move_gripper(client: ModbusSerialClient, slave_id: int, position: int, speed: int, force: int) -> None:
    pos = max(0, min(255, position))
    speed = max(0, min(255, speed))
    force = max(0, min(255, force))

    # 核心修复 1：高低字节翻转。
    # 把 speed 放在高八位，pos 放在低八位，抵消硬件带来的反转错觉。
    reg2 = (speed << 8) | pos
    reg3 = (force << 8) | 0x00

    # 核心修复 2：脉冲触发 (Pulse Trigger)
    # 先发送 rGTO=0，再发送 rGTO=1，强制夹爪状态机复位并接收新位置。
    reg1_idle = 0x0100  # rACT=1, rGTO=0 (暂停)
    reg1_go = 0x0900    # rACT=1, rGTO=1 (执行)

    write_regs(client, 0x03E8, [reg1_idle, reg2, reg3], slave_id)
    time.sleep(0.03)  # 给硬件一点反应时间
    write_regs(client, 0x03E8, [reg1_go, reg2, reg3], slave_id)
    
    print(f"[Robotiq] 指令已发送 -> 位置: {pos:3d}, 速度: {speed}, 力度: {force}")

def main():
    parser = argparse.ArgumentParser(description="Robotiq 极简控制脚本")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="串口设备")
    parser.add_argument("--baudrate", type=int, default=115200, help="波特率")
    parser.add_argument("--slave-id", type=int, default=9, help="Robotiq 从站 ID")
    parser.add_argument("--speed", type=int, default=255, help="移动速度")
    parser.add_argument("--force", type=int, default=150, help="抓取力度")
    args = parser.parse_args()

    client = ModbusSerialClient(port=args.port, baudrate=args.baudrate, timeout=1.0)

    if not client.connect():
        print(f"错误: 无法连接到 {args.port}")
        return

    try:
        activate_gripper(client, args.slave_id)

        print("\n" + "="*35)
        print("  [Robotiq] 交互控制已启动")
        print("  输入 0   : 完全张开")
        print("  输入 255 : 完全闭合")
        print("  输入 q   : 退出程序")
        print("="*35 + "\n")

        while True:
            cmd = input("请输入目标位置 (0-255) 或 'q' 退出: ").strip().lower()
            if cmd == "q":
                break
            elif cmd.isdigit():
                move_gripper(client, args.slave_id, int(cmd), args.speed, args.force)
            else:
                print("[Robotiq] 无效输入，请输入数字！")

    except KeyboardInterrupt:
        print("\n[Robotiq] 程序被手动中断")
    finally:
        client.close()
        print("[Robotiq] 串口已关闭。")

if __name__ == '__main__':
    main()