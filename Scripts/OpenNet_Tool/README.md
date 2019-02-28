
Author     KMS - Martin Dubois, ing.
Copyright  (C) 2019 KMS. All right reserved.
Product    OpenNet
File       Scripts/OpenNet_Tool/README.md

For Windows, the script in this folder are nammed

	{Setup}{Computer}{User}{Kernel}{KernelDbg}{DebugFilter}{GPU_State}_{GPU}_{OSVersion}.txt

	Setup

		A			A dual port network adapter with each port connected to each other

	Computer

		00			Memory			32 GB
					Motherboard		ASUS X99-A II
					Processor		i7-5930K	3.5 GHz

		01			Memory			16 GB
					Motherboard		MSI Z97-G43
					Processor		i7-4790		3.6 GHz

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

		WX9100

	OSVersion

		10			Windows 10

For Linux, the scripts in this folder are named

	{Setup}{Computer}{User}{Kernel}{GPU_State}_{GPU}_{OSVersion}.txt

	Setup

		A			A dual port network adapter with each port connected to each other

	Computer

		00			Memory			32 GB
					Motherboard		ASUS X99-A II
					Processor		i7-5930K	3.5 GHz

		01			Memory			16 GB
					Motherboard		MSI Z97-G43
					Processor		i7-4790		3.6 GHz

	User

		D			Debug

		R			Release

	Kernel

		D			Debug

		R			Release

	GPU_State

		A			Active

		I			Inactive

		U			Used as graphic card

	GPU
	
		K620

	OSVersion

		18.04		Ubuntu 18.04