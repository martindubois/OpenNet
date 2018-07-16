
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/Kernel.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/SourceCode.h>
#include <OpenNet/StatisticsProvider.h>

namespace OpenNet
{

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  The Kernel class
    /// \endcond
    /// \cond fr
    /// \brief  La classe Kernel
    /// \endcond
    class Kernel : public SourceCode, public StatisticsProvider
    {

    public:

        /// \cond en
        /// \brief  Constructor
        /// \endcond
        /// \cond fr
        /// \brief  Constructeur
        /// \endcond
        OPEN_NET_PUBLIC Kernel();

        /// \cond en
        /// \brief  Disable OpenCL profiling
        /// \endcond
        /// \cond fr
        /// \brief  Desactiver le profiling OpenCL
        /// \endcond
        /// \retval STATUS_PROFILING_ALREADY_DISABLED
        OPEN_NET_PUBLIC Status DisableProfiling();

        /// \cond en
        /// \brief  Enable OpenCL profiling
        /// \endcond
        /// \cond fr
        /// \brief  Activer le profiling OpenCL
        /// \endcond
        /// \retval STATUS_PROFILING_ALREADY_ENABLED
        OPEN_NET_PUBLIC Status EnableProfiling();

        /// \cond en
        /// \brief  Retrieve the build log
        /// \return This method returns the address of an internal buffer.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir le log de compilation
        /// \retval Cette methode retourne l'adresse d'un espace de memoire
        ///         interne.
        /// \endcond
        OPEN_NET_PUBLIC const char * GetBuildLog() const;

        /// \cond en
        /// \brief  Retrieve the code line count
        /// \return This method returns the code line cout.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir le nombre de ligne de code
        /// \retval Cette methode retourne le nombre de ligne de code.
        /// \endcond
        OPEN_NET_PUBLIC unsigned int  GetCodeLineCount();

        /// \cond en
        /// \brief  Retrieve the code lines
        /// \return This method returns the address of an internal buffer.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir les lignes de code
        /// \retval Cette methode retourne l'adresse d'un espace de memoire
        ///         interne.
        /// \endcond
        OPEN_NET_PUBLIC const char ** GetCodeLines();

        /// \cond en
        /// \brief  Is the OpenCL profiling enabled?
        /// \endcond
        /// \cond fr
        /// \brief  Est-ce que le profiling OpenCL est active?
        /// \endcond
        /// \retval false
        /// \retval true
        OPEN_NET_PUBLIC bool IsProfilingEnabled() const;

        // ===== SourceCode =================================================
        OPEN_NET_PUBLIC virtual              ~Kernel     ();
        OPEN_NET_PUBLIC virtual Status       AppendCode  (const char * aCode, unsigned int aCodeSize_byte);
        OPEN_NET_PUBLIC virtual Status       ResetCode   ();
        OPEN_NET_PUBLIC virtual Status       SetCode     (const char * aFileName);
        OPEN_NET_PUBLIC virtual Status       SetCode     (const char * aCode, unsigned int aCodeSize_byte);
        OPEN_NET_PUBLIC virtual Status       Display     (FILE * aOut) const;
        OPEN_NET_PUBLIC virtual unsigned int Edit_Remove (const char * aSearch);
        OPEN_NET_PUBLIC virtual unsigned int Edit_Replace(const char * aSearch, const char * aReplace);

        // ===== StatisticsProvider =========================================
        virtual Status GetStatistics  (unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset);
        virtual Status ResetStatistics();

    // Internal:

        // TODO  Include.OpenNet.Kernel
        //       Deplacer la definition de BUILD_LOG_MAX_SIZE_byte dans un
        //       fichier prive.

        static const unsigned int BUILD_LOG_MAX_SIZE_byte;

        void AddStatistics(uint64_t aQueued, uint64_t aSubmit, uint64_t aStart, uint64_t aEnd);

        void * AllocateBuildLog();

    private:

        Kernel(const Kernel &);

        const Kernel & operator = (const Kernel &);

        void CodeLines_Count   ();
        void CodeLines_Generate();

        void Invalidate();

        char         * mBuildLog         ;
        char         * mCodeLineBuffer   ;
        unsigned int   mCodeLineCount    ;
        const char  ** mCodeLines        ;
        bool           mProfilingEnabled ;
        unsigned int * mStatistics       ;
        uint64_t       mStatisticsSums[3];

    };

}