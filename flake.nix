{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/master";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShells = {
          default = pkgs.mkShell {
            nativeBuildInputs = with pkgs; [
              act
              ffmpeg-full
              go-task
              lsyncd
              parallel
              poetry
              pre-commit
              pv
              pyright
              python312
              rsync
              ruff
              websocat
              uv
            ];
            shellHook = ''
              poetry env use python3.12
              source $(poetry env info --path)/bin/activate
              export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.zlib}/lib:$LD_LIBRARY_PATH
              source .env
            '';
          };
        };
        formatter = pkgs.nixfmt;
      }
    );
}
