/*++

Module Name:

    public.h

Abstract:

    This module contains the common declarations shared by driver
    and user applications.

Environment:

    user and kernel

--*/

//
// Define an Interface Guid so that app can find the device and talk to it.
//

DEFINE_GUID (GUID_DEVINTERFACE_ONKPro1000,
    0xbf94dbe4,0xd0e3,0x4782,0xaf,0x00,0xfd,0xec,0x26,0x96,0xa6,0x8a);
// {bf94dbe4-d0e3-4782-af00-fdec2696a68a}
