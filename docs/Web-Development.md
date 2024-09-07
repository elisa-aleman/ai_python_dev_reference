# Web development notes

TODO:
- [ ] Re-check old narrative
- [ ] Check all syntax highlights and location comments
- [ ] Add style guide for location comments
- [ ] Wrap all links
- [ ] Write about Python FastAPI to replace old Flask docs


## Python FastAPI

Under construction

## nodenv for Node.js tools

Sources:
- [anyenv: nodenv](https://github.com/anyenv/anyenv)
- [nodenv](https://github.com/nodenv/nodenv)

[nodenv](https://github.com/nodenv/nodenv) allows you to install isolated Node.js environments.

I mainly don't do front end development or use Node.js at all for my personal projects, but nodenv allows me to install other JS based tools I've found useful, like [Mermaid](https://mermaid.js.org/).

```sh
# @ shell(linux/mac_osx/wsl)

anyenv install nodenv
exec $SHELL -l
```

Or, if you didn't install `anyenv`, follow the [github based installation](https://github.com/nodenv/nodenv?tab=readme-ov-file#basic-github-checkout).

Now install the latest version (I'm not knowledgeable enough about compatibility issues yet to determine if older versions are better)
```sh
# @ shell(linux/mac_osx/wsl)

nodenv install -l
```

At the time of writing (2024-09-07):

```sh
# @ shell(linux/mac_osx/wsl)
nodenv install 22.8.0
nodenv global 22.8.0
```

Test if the `npm` commands work:
```sh
# @ shell(linux/mac_osx/wsl)
npm -h

```

## Mermaid CLI

There's a Node.js command line tool called [`mermaid-cli`](https://github.com/mermaid-js/mermaid-cli) which is necessary for the Mermaid Sublime plugin. However, I ran into dependency issues.

```
npm warn deprecated puppeteer@19.11.1: < 22.8.2 is no longer supported
```

There's already a [pull request fixing this](https://github.com/mermaid-js/mermaid-cli/pull/739), but hasn't been merged at the time of writing this (2024-09-07)

## Ruby, Bundler and Jekyll for static websites

Sometimes it might be necessary to present results in HTML websites. I also make websites on my free time, and lots of researchers have their projects on a github pages website. For this, I like to use Jekyll in combination with github pages.

For this, I like to use Jekyll, which needs Ruby and Bundler. 

### Install Ruby on Windows

First lets install Ruby, on which Jekyll and Bundler will be later installed.

- Reference:
    - [Jekyll installation guide for windows](https://jekyllrb.com/docs/installation/windows/)

Install Ruby+Devkit on Windows with [RubyInstaller](https://rubyinstaller.org/)

- Don't forget to select the add Ruby related items to PATH option
- Run the ridk install at the last stage of the installation wizard.
- You'll be prompted to install MSYS2, I selected option 3.

Open a new command prompt window from the start menu, so that changes to the PATH environment variable becomes effective. 
`echo %PATH%` on cmd and `echo $PATH` on Git Bash should both have Ruby related paths now.


You can do this on the Git Bash too, but the output will be delayed, so run on cmd.

### Install Ruby on MacOSX

- Reference:
    - [Jekyll installation guide for mac](https://jekyllrb.com/docs/installation/macos/)

First lets install Ruby, on which Jekyll and Bundler will be later installed.

```
brew install ruby
```

Then, we have to add to the $PATH so that ruby gems are found:

```
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zprofile
echo 'export PATH="/usr/local/opt/ruby/bin:$PATH"' >> ~/.zprofile
echo 'export PATH="~/.local/share/gem/ruby/X.X.X/bin:$PATH"' >> ~/.zprofile
source ~/.zprofile
```

Where `X.X.X ` is the version you have installed.

In my case it was:

```
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zprofile
echo 'export PATH="/usr/local/opt/ruby/bin:$PATH"' >> ~/.zprofile
echo 'export PATH="~/.local/share/gem/ruby/3.0.0/bin:$PATH"' >> ~/.zprofile
source ~/.zprofile
```

Check that we're indeed using the Homebrew version of ruby:

```
which ruby
```
Which should output this:
```
/usr/local/opt/ruby/bin/ruby
```


### Install Ruby on Linux

- Reference:
    - [Jekyll installation guide for Ubuntu](https://jekyllrb.com/docs/installation/ubuntu/)

First lets install Ruby, on which Jekyll and Bundler will be later installed.

```
sudo apt-get install ruby-full build-essential zlib1g-dev
```

Then, we have to add to the $PATH so that ruby gems are found:

```
echo '# Install Ruby Gems to ~/gems' >> ~/.bash_profile
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bash_profile
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bash_profile
source ~/.bash_profile
```

### Install Jekyll and Bundler

Install Jekyll and Bundler:

```
gem install bundler jekyll jekyll-sitemap
```

For MacOSX, it is necessary to install at the user level, if I recall correctly...

```
gem install --user-install bundler jekyll jekyll-sitemap
```

Check if installed properly `jekyll -v` on cmd or git bash.

Now it's installed! I'll be using this for local websites, but we're going to follow a tutorial on how to make Jekyll Github Pages, so even though we're not using GitHub, give this a read.

https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll

Make a new repository for your website on local git.

Now, once you have your webiste repository and you're ready to test the jekyll serve, do the following:

```
cd (your_repository_here)
bundle init
bundle add jekyll
bundle add jekyll-sitemap
bundle add webrick
```

And then all that's left to do is to serve the website with jekyll!
Also for the sitemaps make sure to check this tutorial:

https://github.com/jekyll/jekyll-sitemap

And add this to your `_config.yml`
```
url: "https://example.com" # the base hostname & protocol for your site
plugins:
  - jekyll-sitemap
```

```
bundle exec jekyll serve
```

If you get an error like:
```
Could not find webrick-1.7.0 in any of the sources
Run `bundle install` to install missing gems.
```

Do as it says and just run:
```
bundle install
```

Now you can work on the website and look at how it changes on screen.

By the way, if you are hosting on GitHub Pages and have a custom domain, you need to add these to the DNS

```
Type    Name    Points to               TTL
a       @       185.199.108.153         600 seconds
a       @       185.199.109.153         600 seconds
a       @       185.199.110.153         600 seconds
a       @       185.199.111.153         600 seconds
cname   www     your-username.github.io 600 seconds   
```
