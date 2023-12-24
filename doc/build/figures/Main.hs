import Control.Monad
import Data.Functor
import Data.List.Extra
import System.Directory

main = do
  files <-
    listDirectory "."
      <&> filter
        ( \x ->
            ".png" `isSuffixOf` x
              || ".svg" `isSuffixOf` x
              || ".jpg" `isSuffixOf` x
        )
  mapM_
    ( \file -> do
        Control.Monad.when ("2022-11-08" `isInfixOf` file) $
          let
            newName = replace "2022-11-08" "2023-11-28" file
           in
            renameFile file newName
    )
    files
