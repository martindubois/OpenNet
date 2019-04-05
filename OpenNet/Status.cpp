
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Status.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes/OpenNet ===================================================
#include <OpenNet/Status.h>

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    const char * mName       ;
    const char * mDescription;
}
StatusInfo;

// Constants
/////////////////////////////////////////////////////////////////////////////

static const StatusInfo STATUS_INFO[OpenNet::STATUS_QTY] =
{
    { "OK", "Success" },

    { "ADAPTER_ALREADY_CONNECTED" , "The adapter is already connected"         },
    { "ADAPTER_ALREADY_SET"       , "The adapter is already set"               },
    { "ADAPTER_NOT_CONNECTED"     , "The adapter is not connected"             },
    { "ADAPTER_NOT_SET"           , "The adapter is not set"                   },
    { "ADAPTER_RUNNING"           , "The adapter is running"                   },
    { "BUFFER_ALLOCATED"          , "At least one buffer is allocated"         },
    { "BUFFER_TOO_SMALL"          , "The buffer is too small"                  },
    { "CANNOT_OPEN_INPUT_FILE"    , "Cannot open input file"                   },
    { "CANNOT_READ_INPUT_FILE"    , "Cannor read input file"                   },
    { "CODE_ALREADY_SET"          , "The code is already set"                  },
    { "CODE_NOT_SET"              , "The code is not set"                      },
    { "CORRUPTED_DRIVER_DATA"     , "Corrupted driver data"                    },
    { "DESTINATION_ALREADY_SET"   , "The destination is already set"           },
    { "DESTINATION_NOT_SET"       , "The destination is not set"               },
    { "EMPTY_CODE"                , "The code is empty"                        },
    { "EMPTY_INPUT_FILE"          , "The input file is empty"                  },
    { "ERROR_CLOSING_FILE"        , "Error closing file"                       },
    { "ERROR_READING_INPUT_FILE"  , "Error reading input file"                 },
    { "EXCEPTION"                 , "Exception"                                },
    { "FILTER_ALREADY_SET"        , "The filter is already set"                },
    { "FILTER_NOT_SET"            , "The filter is not set"                    },
    { "FILTER_SET"                , "The filter is set"                        },
    { "INPUT_FILE_TOO_LARGE"      , "The input file is too large"              },
    { "INTERNAL_ERROR"            , "Internal error"                           },
    { "INVALID_ADAPTER"           , "Invalid adapter"                          },
    { "INVALID_ARGUMENT_COUNT"    , "Invalid argument count"                   },
    { "INVALID_BANDWIDTH"         , "Invalid bandwidth"                        },
    { "INVALID_BUTTON_INDEX"      , "Invalid button index"                     },
    { "INVALID_BUFFER_COUNT"      , "Invalid buffer count"                     },
    { "INVALID_COMMAND_INDEX"     , "Invalid command index"                    },
    { "INVALID_INDEX"             , "Invalid index"                            },
    { "INVALID_LINK_SPEED"        , "Invalid link speed"                       },
    { "INVALID_MODE"              , "Invalid mode"                             },
    { "INVALID_OFFSET"            , "Invalid offset"                           },
    { "INVALID_PACKET_SIZE"       , "Invalid packet size"                      },
    { "INVALID_PAGE_INDEX"        , "Invalid page index"                       },
    { "INVALID_PROCESSOR"         , "Invalid processor"                        },
    { "INVALID_PROTOCOL"          , "Invalid protocol"                         },
    { "INVALID_REFERENCE"         , "Invalid reference"                        },
    { "INVALID_SIZE"              , "Invalid size"                             },
    { "IOCTL_ERROR"               , "IoCtl error"                              },
    { "NAME_TOO_LONG"             , "The name is too long"                     },
    { "NAME_TOO_SHORT"            , "The name it too short"                    },
    { "NO_ADAPTER_CONNECTED"      , "No adapter connected"                     },
    { "NO_BUFFER"                 , "No buffer"                                },
    { "NO_DESTINATION_SET"        , "No destination set"                       },
    { "NOT_ADMINISTRATOR"         , "The process does not run as administrator"},
    { "NOT_ALLOWED_NULL_ARGUMENT" , "Not allowed NULL argument"                },
    { "NOT_IMPLEMENTED"           , "Not implemented"                          },
    { "OPEN_CL_ERROR"             , "OpenCL error"                             },
    { "PACKET_GENERATOR_RUNNING"  , "The packet generator is running"          },
    { "PACKET_GENERATOR_STOPPED"  , "The packet generator is stopped"          },
    { "PACKET_TOO_LARGE"          , "The packet is too large"                  },
    { "PACKET_TOO_SMALL"          , "The packet is too small"                  },
    { "PROCESSOR_ALREADY_SET"     , "The processor is already set"             },
    { "PROCESSOR_NOT_SET"         , "The processor is not set"                 },
    { "PROFILING_ALREADY_DISABLED", "The profiling is already disabled"        },
    { "PROFILING_ALREADY_ENABLED" , "The profiling is already enabled"         },
    { "REBOOT_REQUIRED"           , "A reboot is required"                     },
    { "SAME_VALUE"                , "Set to the same value"                    },
    { "SYSTEM_ALREADY_STARTED"    , "The system is already started"            },
    { "SYSTEM_RUNNING"            , "The system is running"                    },
    { "SYSTEM_NOT_STARTED"        , "The system is not started"                },
    { "THREAD_CREATE_ERROR"       , "The thread creation reported and error"   },
    { "THREAD_CLOSE_ERROR"        , "The thread closure reported and error"    },
    { "THREAD_STOP_TIMEOUT"       , "The thread did not stop in allowed time"  },
    { "THREAD_TERMINATE_ERROR"    , "The thread termination reported an error" },
    { "TOO_MANY_BUFFER"           , "Too many buffer allocated"                },
};

namespace OpenNet
{

    // Functions
    /////////////////////////////////////////////////////////////////////////

    const char * Status_GetDescription(Status aStatus)
    {
        if (STATUS_QTY <= aStatus)
        {
            return "Invalid status code";
        }

        return STATUS_INFO[aStatus].mDescription;
    }

    const char * Status_GetName(Status aStatus)
    {
        if (STATUS_QTY <= aStatus)
        {
            return "INVALID_STATUS_CODE";
        }

        return STATUS_INFO[aStatus].mName;
    }

    Status Status_Display(Status aStatus, FILE * aOut)
    {
        if (NULL == aOut)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        fprintf(aOut, "%u = %s - %s\n", aStatus, Status_GetName(aStatus), Status_GetDescription(aStatus));

        return STATUS_OK;
    }

}
