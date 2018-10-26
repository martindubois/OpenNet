
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/Kernel_Forward.h
/// \brief   OpenNet::Kernel_Forward

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Kernel.h>

namespace OpenNet
{
    class Adapter;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  The Kernel_Forward class
    /// \endcond
    /// \cond fr
    /// \brief  La classe Kernel_Forward
    /// \endcond
    class Kernel_Forward : public Kernel
    {

    public:

        /// \cond en
        /// \brief  Constructor
        /// \endcond
        /// \cond fr
        /// \brief  Constructeur
        /// \endcond
        OPEN_NET_PUBLIC Kernel_Forward();

        /// \cond en
        /// \brief  Add a destination adapter
        /// \param  aAdapter [---;RW-] The Adapter
        /// \endcond
        /// \cond fr
        /// \brief  Ajouter un Adapter de destination
        /// \param  aAdapter [---;RW-] L'Adapter
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_DESTINATION_ALREADY_SET
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        OPEN_NET_PUBLIC Status AddDestination(Adapter * aAdapter);

        /// \cond en
        /// \brief  Remove a destination adapter
        /// \param  aAdapter [---;RW-] The Adapter
        /// \endcond
        /// \cond fr
        /// \brief  Retirer un Adapter de destination
        /// \param  aAdapter [---;RW-] L'Adapter
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_DESTINATION_NOT_SET
        /// \retval STATUS_NO_DESTINATION_SET
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        OPEN_NET_PUBLIC Status RemoveDestination(Adapter * aAdapter);

        /// \cond en
        /// \brief  Remove all destination Adapter
        /// \endcond
        /// \cond fr
        /// \brief  Retirer tous les Adapter de destination
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NO_DESTINATION_NOT
        OPEN_NET_PUBLIC Status ResetDestinations();

        // ===== SourceCode =====================================================
        OPEN_NET_PUBLIC virtual        ~Kernel_Forward();
        OPEN_NET_PUBLIC virtual Status Display        (FILE * aOut) const;


    private:

        Kernel_Forward(const Kernel_Forward &);

        const Kernel_Forward & operator = (const Kernel_Forward &);

        void GenerateCode();

        uint32_t mDestinations;

    };

}
