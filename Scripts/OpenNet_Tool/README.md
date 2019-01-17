
Author     KMS - Martin Dubois, ing.
Copyright  (C) 2019 KMS. All right reserved.
Product    OpenNet
File       Scripts/OpenNet_Tool/README.md

The script in this folder are nammed

	{Setup}{Computer}{User}{Kernel}{KernelDbg}{DebugFilter}{GPU_State}_{GPU}_{OSVersion}.txt

	Setup

		A			A dual port network adapter with each port connected to each other

	Computer

		00			Memory			32 GB
					Motherboard		ASUS X99
					Processor		i7-5930K	3.5 GHz

	User

		D			Debug

		R			Release

	Kernel

		D			Debug

		R			Release

	KernelDbg

		C			Connected

		N			Not connected

	DebugFilter

		00000000	The value of
					HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Debug Print Filter\IHVDRIVER
					in hex format

	GPU_State

		A			Active
		I			Inactive
		U			Used as graphic card

	GPU
	
		WX5100

	OSVersion

		10			Windows 10

		18.04		Ubuntu 18.04
