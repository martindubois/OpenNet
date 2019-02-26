
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/Regs.h

#pragma once

// Macro
/////////////////////////////////////////////////////////////////////////////

#define REG_RESERVED(A,E) uint32_t mReserved_##A[(0x##E - 0x##A) / 4]
