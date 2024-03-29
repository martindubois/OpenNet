
; Author     KMS - Martin Dubois, ing.
; Copyright  (C) 2019 KMS. All rights reserved.
; Product    OpenNet
; File       OpenNet.iss

; CODE REVIEW  2019-07-26  KMS - Martin Dubois, ing.

[Setup]
AllowNetworkDrive=no
AllowUNCPath=no
AppCopyright=Copyright (C) 2019 KMS.
AppName=OpenNet
AppPublisher=KMS
AppPublisherURL=http://www.kms-quebec.com
AppSupportURL=http://www.kms-quebec.com
AppVersion=1.0.12
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
DefaultDirName={pf}\OpenNet
LicenseFile=License.txt
MinVersion=10.0
OutputBaseFilename=OpenNet_1.0.12
OutputDir=Installer

[Files]
Source: "_DocUser\OpenNet.ReadMe.txt"                            ; DestDir: "{app}"; Flags: isreadme
Source: "Includes\OpenNet\Adapter.h"                             ; DestDir: "{app}\Includes\OpenNet"
Source: "Includes\OpenNet\Buffer.h"                              ; DestDir: "{app}\Includes\OpenNet"
Source: "Includes\OpenNet\Function.h"                            ; DestDir: "{app}\Includes\OpenNet"
Source: "Includes\OpenNet\Function_Forward.h"                    ; DestDir: "{app}\Includes\OpenNet"
Source: "Includes\OpenNet\Kernel.h"                              ; DestDir: "{app}\Includes\OpenNet"
Source: "Includes\OpenNet\Kernel_Forward.h"                      ; DestDir: "{app}\Includes\OpenNet"
Source: "Includes\OpenNet\OpenNet.h"                             ; DestDir: "{app}\Includes\OpenNet"
Source: "Includes\OpenNet\PacketGenerator.h"                     ; DestDir: "{app}\Includes\OpenNet"
Source: "Includes\OpenNet\Processor.h"                           ; DestDir: "{app}\Includes\OpenNet"
Source: "Includes\OpenNet\SetupTool.h"                           ; DestDir: "{app}\Includes\OpenNet"
Source: "Includes\OpenNet\SourceCode.h"                          ; DestDir: "{app}\Includes\OpenNet"
Source: "Includes\OpenNet\StatisticsProvider.h"                  ; DestDir: "{app}\Includes\OpenNet"
Source: "Includes\OpenNet\Status.h"                              ; DestDir: "{app}\Includes\OpenNet"
Source: "Includes\OpenNet\System.h"                              ; DestDir: "{app}\Includes\OpenNet"
Source: "Includes\OpenNet\UserBuffer.h"                          ; DestDir: "{app}\Includes\OpenNet"
Source: "Includes\OpenNetK\Adapter_Types.h"                      ; DestDir: "{app}\Includes\OpenNetK"
Source: "Includes\OpenNetK\ARP.h"                                ; DestDir: "{app}\Includes\OpenNetK"
Source: "Includes\OpenNetK\ByteOrder.h"                          ; DestDir: "{app}\Includes\OpenNetK"
Source: "Includes\OpenNetK\Ethernet.h"                           ; DestDir: "{app}\Includes\OpenNetK"
Source: "Includes\OpenNetK\IPv4.h"                               ; DestDir: "{app}\Includes\OpenNetK"
Source: "Includes\OpenNetK\IPv6.h"                               ; DestDir: "{app}\Includes\OpenNetK"
Source: "Includes\OpenNetK\Kernel.h"                             ; DestDir: "{app}\Includes\OpenNetK"
Source: "Includes\OpenNetK\Kernel_OpenCL.h"                      ; DestDir: "{app}\Includes\OpenNetK"
Source: "Includes\OpenNetK\PacketGenerator_Types.h"              ; DestDir: "{app}\Includes\OpenNetK"
Source: "Includes\OpenNetK\RegEx.h"                              ; DestDir: "{app}\Includes\OpenNetK"
Source: "Includes\OpenNetK\TCP.h"                                ; DestDir: "{app}\Includes\OpenNetK"
Source: "Includes\OpenNetK\Types.h"                              ; DestDir: "{app}\Includes\OpenNetK"
Source: "Includes\OpenNetK\UDP.h"                                ; DestDir: "{app}\Includes\OpenNetK"
Source: "ONK_Pro1000\_DocUser\OpenNet.ONK_Hardware.ReadMe.txt"   ; DestDir: "{app}"
Source: "OpenNet\_DocUser\OpenNet.OpenNet.ReadMe.txt"            ; DestDir: "{app}"
Source: "OpenNet_Setup\_DocUser\OpenNet.OpenNet_Setup.ReadMe.txt"; DestDir: "{app}"
Source: "OpenNet_Tool\_DocUser\OpenNet.OpenNet_Tool.ReadMe.txt"  ; DestDir: "{app}"
Source: "x64\Release\ONK_Pro1000\onk_pro1000.cat"                ; DestDir: "{app}\Drivers\ONK_Hardware"
Source: "x64\Release\ONK_Pro1000\ONK_Pro1000.inf"                ; DestDir: "{app}\Drivers\ONK_Hardware"
Source: "x64\Release\ONK_Pro1000\ONK_Pro1000.sys"                ; DestDir: "{app}\Drivers\ONK_Hardware"
Source: "x64\Release\OpenNet.dll"                                ; DestDir: "{app}"
Source: "x64\Release\OpenNet.lib"                                ; DestDir: "{app}\Libraries"
Source: "x64\Release\OpenNet_Setup.exe"                          ; DestDir: "{app}"
Source: "x64\Release\OpenNet_Tool.exe"                           ; DestDir: "{app}"

[Run]
Filename: "{app}\OpenNet_Setup.exe"; Parameters: "install"

[UninstallRun]
Filename: "{app}\OpenNet_Setup.exe"; Parameters: "uninstall"
