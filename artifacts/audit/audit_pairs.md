# False-negative audit — 100 pairs (50 per condition, blind)

seed 13, version `phase01-v6`. For each pair decide whether the two fragments implement the **same functionality** (BigCloneBench's clone definition). Write `clone`, `not_clone` or `unsure` in `audit_labels.csv`; do not open `audit_key.csv` until you run `score`.

Rubric: `clone` = same functionality whatever the syntax (Type-1..4, incl. weak T3/T4); `not_clone` = different functionality or one is only a small part of the other; `unsure` = cannot tell in about a minute.

## P001

**A**

```java
public static void copyFile(File destFile, File src) throws IOException {
        File destDir = destFile.getParentFile();
        File tempFile = new File(destFile + "_tmp");
        destDir.mkdirs();
        InputStream is = new FileInputStream(src);
        try {
            FileOutputStream os = new FileOutputStream(tempFile);
            try {
                byte[] buf = new byte[8192];
                int len;
                while ((len = is.read(buf)) > 0) os.write(buf, 0, len);
            } finally {
                os.close();
            }
        } finally {
            is.close();
        }
        destFile.delete();
        if (!tempFile.renameTo(destFile)) throw new IOException("Unable to rename " + tempFile + " to " + destFile);
    }
```

**B**

```java
public static void copyFile(File src, File dest, int bufSize, boolean force) throws IOException {
        if (dest.exists()) {
            if (force) {
                dest.delete();
            } else {
                throw new IOException(className + "Cannot overwrite existing file: " + dest.getAbsolutePath());
            }
        }
        byte[] buffer = new byte[bufSize];
        int read = 0;
        InputStream in = null;
        OutputStream out = null;
        try {
            in = new FileInputStream(src);
            out = new FileOutputStream(dest);
            while (true) {
                read = in.read(buffer);
                if (read == -1) {
                    break;
                }
                out.write(buffer, 0, read);
            }
        } finally {
            if (in != null) {
                try {
                    in.close();
                } finally {
                    if (out != null) {
                        out.close();
                    }
                }
            }
        }
    }
```

## P002

**A**

```java
public void copyFile(File a_fileSrc, File a_fileDest, boolean a_append) throws IOException {
        a_fileDest.getParentFile().mkdirs();
        FileInputStream in = null;
        FileOutputStream out = null;
        FileChannel fcin = null;
        FileChannel fcout = null;
        try {
            in = new FileInputStream(a_fileSrc);
            out = new FileOutputStream(a_fileDest, a_append);
            fcin = in.getChannel();
            fcout = out.getChannel();
            ByteBuffer buffer = ByteBuffer.allocate(16 * 1024);
            while (true) {
                buffer.clear();
                int r = fcin.read(buffer);
                if (r == -1) {
                    break;
                }
                buffer.flip();
                fcout.write(buffer);
            }
        } catch (IOException ex) {
            throw ex;
        } finally {
            if (in != null) {
                in.close();
            }
            if (out != null) {
                out.close();
            }
            if (fcin != null) {
                fcin.close();
            }
            if (fcout != null) {
                fcout.close();
            }
        }
    }
```

**B**

```java
public static boolean downloadFile(String url, String destination) {
        BufferedInputStream bi = null;
        BufferedOutputStream bo = null;
        File destfile;
        try {
            java.net.URL fileurl;
            try {
                fileurl = new java.net.URL(url);
            } catch (MalformedURLException e) {
                return false;
            }
            bi = new BufferedInputStream(fileurl.openStream());
            destfile = new File(destination);
            if (!destfile.createNewFile()) {
                destfile.delete();
                destfile.createNewFile();
            }
            bo = new BufferedOutputStream(new FileOutputStream(destfile));
            int readedbyte;
            while ((readedbyte = bi.read()) != -1) {
                bo.write(readedbyte);
            }
            bo.flush();
        } catch (IOException ex) {
            return false;
        } finally {
            try {
                bi.close();
                bo.close();
            } catch (Exception ex) {
            }
        }
        return true;
    }
```

## P003

**A**

```java
public void test() throws Exception {
        StorageString s = new StorageString("UTF-8");
        s.addText("Test");
        try {
            s.getOutputStream();
            fail("Should throw IOException as method not supported.");
        } catch (IOException e) {
        }
        try {
            s.getWriter();
            fail("Should throw IOException as method not supported.");
        } catch (IOException e) {
        }
        s.addText("ing is important");
        s.close(ResponseStateOk.getInstance());
        assertEquals("Testing is important", s.getText());
        InputStream input = s.getInputStream();
        StringWriter writer = new StringWriter();
        IOUtils.copy(input, writer, "UTF-8");
        assertEquals("Testing is important", writer.toString());
    }
```

**B**

```java
public void testHandler() throws MalformedURLException, IOException {
        assertTrue("This test can only be run once in a single JVM", imageHasNotBeenInstalledInThisJVM);
        URL url;
        Handler.installImageUrlHandler((ImageSource) new ClassPathXmlApplicationContext("org/springframework/richclient/image/application-context.xml").getBean("imageSource"));
        try {
            url = new URL("image:test");
            imageHasNotBeenInstalledInThisJVM = false;
        } catch (MalformedURLException e) {
            fail("protocol was not installed");
        }
        url = new URL("image:image.that.does.not.exist");
        try {
            url.openConnection();
            fail();
        } catch (NoSuchImageResourceException e) {
        }
        url = new URL("image:test.image.key");
        url.openConnection();
    }
```

## P004

**A**

```java
private void weightAndPlaceClasses() {
        int rows = getRows();
        for (int curRow = _maxPackageRank; curRow < rows; curRow++) {
            xPos = getHGap() / 2;
            BOTLObjectSourceDiagramNode[] rowObject = getObjectsInRow(curRow);
            for (int i = 0; i < rowObject.length; i++) {
                if (curRow == _maxPackageRank) {
                    int nDownlinks = rowObject[i].getDownlinks().size();
                    rowObject[i].setWeight((nDownlinks > 0) ? (1 / nDownlinks) : 2);
                } else {
                    Vector uplinks = rowObject[i].getUplinks();
                    int nUplinks = uplinks.size();
                    if (nUplinks > 0) {
                        float average_col = 0;
                        for (int j = 0; j < uplinks.size(); j++) {
                            average_col += ((BOTLObjectSourceDiagramNode) (uplinks.elementAt(j))).getColumn();
                        }
                        average_col /= nUplinks;
                        rowObject[i].setWeight(average_col);
                    } else {
                        rowObject[i].setWeight(1000);
                    }
                }
            }
            int[] pos = new int[rowObject.length];
            for (int i = 0; i < pos.length; i++) {
                pos[i] = i;
            }
            boolean swapped = true;
            while (swapped) {
                swapped = false;
                for (int i = 0; i < pos.length - 1; i++) {
                    if (rowObject[pos[i]].getWeight() > rowObject[pos[i + 1]].getWeight()) {
                        int temp = pos[i];
                        pos[i] = pos[i + 1];
                        pos[i + 1] = temp;
                        swapped = true;
                    }
                }
            }
            for (int i = 0; i < pos.length; i++) {
                rowObject[pos[i]].setColumn(i);
                if ((i > _vMax) && (rowObject[pos[i]].getUplinks().size() == 0) && (rowObject[pos[i]].getDownlinks().size() == 0)) {
                    if (getColumns(rows - 1) > _vMax) {
                        rows++;
                    }
                    rowObject[pos[i]].setRank(rows - 1);
                } else {
                    rowObject[pos[i]].setLocation(new Point(xPos, yPos));
                    xPos += rowObject[pos[i]].getSize().getWidth() + getHGap();
                }
            }
            yPos += getRowHeight(curRow) + getVGap();
        }
    }
```

**B**

```java
public String getSummaryText() {
        if (summaryText == null) {
            for (Iterator iter = xdcSources.values().iterator(); iter.hasNext(); ) {
                XdcSource source = (XdcSource) iter.next();
                File packageFile = new File(source.getFile().getParentFile(), "xdc-package.html");
                if (packageFile.exists()) {
                    Reader in = null;
                    try {
                        in = new FileReader(packageFile);
                        StringWriter out = new StringWriter();
                        IOUtils.copy(in, out);
                        StringBuffer buf = out.getBuffer();
                        int pos1 = buf.indexOf("<body>");
                        int pos2 = buf.lastIndexOf("</body>");
                        if (pos1 >= 0 && pos1 < pos2) {
                            summaryText = buf.substring(pos1 + 6, pos2);
                        } else {
                            summaryText = "";
                        }
                    } catch (FileNotFoundException e) {
                        LOG.error(e.getMessage(), e);
                        summaryText = "";
                    } catch (IOException e) {
                        LOG.error(e.getMessage(), e);
                        summaryText = "";
                    } finally {
                        if (in != null) {
                            try {
                                in.close();
                            } catch (IOException e) {
                                LOG.error(e.getMessage(), e);
                            }
                        }
                    }
                    break;
                } else {
                    summaryText = "";
                }
            }
        }
        return summaryText;
    }
```

## P005

**A**

```java
public void copyFilesIntoProject(HashMap<String, String> files) {
        Set<String> filenames = files.keySet();
        for (String key : filenames) {
            String realPath = files.get(key);
            if (key.equals("fw4ex.xml")) {
                try {
                    FileReader in = new FileReader(new File(realPath));
                    FileWriter out = new FileWriter(new File(project.getLocation() + "/" + bundle.getString("Stem") + STEM_FILE_EXETENSION));
                    int c;
                    while ((c = in.read()) != -1) out.write(c);
                    in.close();
                    out.close();
                } catch (FileNotFoundException e) {
                    Activator.getDefault().showMessage("File " + key + " not found... Error while moving files to the new project.");
                } catch (IOException ie) {
                    Activator.getDefault().showMessage("Error while moving " + key + " to the new project.");
                }
            } else {
                try {
                    FileReader in = new FileReader(new File(realPath));
                    FileWriter out = new FileWriter(new File(project.getLocation() + "/" + key));
                    int c;
                    while ((c = in.read()) != -1) out.write(c);
                    in.close();
                    out.close();
                } catch (FileNotFoundException e) {
                    Activator.getDefault().showMessage("File " + key + " not found... Error while moving files to the new project.");
                } catch (IOException ie) {
                    Activator.getDefault().showMessage("Error while moving " + key + " to the new project.");
                }
            }
        }
    }
```

**B**

```java
private void regattaBackup() {
        SwingWorker sw = new SwingWorker() {

            Regatta lRegatta = fRegatta;

            public Object construct() {
                String fullName = lRegatta.getSaveDirectory() + lRegatta.getSaveName();
                System.out.println(MessageFormat.format(res.getString("MainMessageBackingUp"), new Object[] { fullName + BAK }));
                try {
                    FileInputStream fis = new FileInputStream(new File(fullName));
                    FileOutputStream fos = new FileOutputStream(new File(fullName + BAK));
                    int bufsize = 1024;
                    byte[] buffer = new byte[bufsize];
                    int n = 0;
                    while ((n = fis.read(buffer, 0, bufsize)) >= 0) fos.write(buffer, 0, n);
                    fos.flush();
                    fos.close();
                } catch (java.io.IOException ex) {
                    Util.showError(ex, true);
                }
                return null;
            }
        };
        sw.start();
    }
```

## P006

**A**

```java
@Override
            public void handle(String s, HttpServletRequest httpServletRequest, HttpServletResponse httpServletResponse, int i) throws IOException, ServletException {
                expected = new StringBuilder();
                System.out.println("uri: " + httpServletRequest.getRequestURI());
                System.out.println("queryString: " + (queryString = httpServletRequest.getQueryString()));
                System.out.println("method: " + httpServletRequest.getMethod());
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                IOUtils.copy(httpServletRequest.getInputStream(), baos);
                System.out.println("body: " + (body = baos.toString()));
                PrintWriter writer = httpServletResponse.getWriter();
                writer.append("testsvar");
                expected.append("testsvar");
                Random r = new Random();
                for (int j = 0; j < 10; j++) {
                    int value = r.nextInt(Integer.MAX_VALUE);
                    writer.append(value + "");
                    expected.append(value);
                }
                System.out.println();
                writer.close();
                httpServletResponse.setStatus(HttpServletResponse.SC_OK);
            }
```

**B**

```java
@Before
    public void setUp() throws Exception {
        configureSslSocketConnector();
        SecurityHandler securityHandler = createBasicAuthenticationSecurityHandler();
        HandlerList handlerList = new HandlerList();
        handlerList.addHandler(securityHandler);
        handlerList.addHandler(new AbstractHandler() {

            @Override
            public void handle(String s, HttpServletRequest httpServletRequest, HttpServletResponse httpServletResponse, int i) throws IOException, ServletException {
                expected = new StringBuilder();
                System.out.println("uri: " + httpServletRequest.getRequestURI());
                System.out.println("queryString: " + (queryString = httpServletRequest.getQueryString()));
                System.out.println("method: " + httpServletRequest.getMethod());
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                IOUtils.copy(httpServletRequest.getInputStream(), baos);
                System.out.println("body: " + (body = baos.toString()));
                PrintWriter writer = httpServletResponse.getWriter();
                writer.append("testsvar");
                expected.append("testsvar");
                Random r = new Random();
                for (int j = 0; j < 10; j++) {
                    int value = r.nextInt(Integer.MAX_VALUE);
                    writer.append(value + "");
                    expected.append(value);
                }
                System.out.println();
                writer.close();
                httpServletResponse.setStatus(HttpServletResponse.SC_OK);
            }
        });
        server.addHandler(handlerList);
        server.start();
    }
```

## P007

**A**

```java
public static final void parse(String infile, String outfile) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(infile));
        DataOutputStream output = new DataOutputStream(new FileOutputStream(outfile));
        int w = Integer.parseInt(reader.readLine());
        int h = Integer.parseInt(reader.readLine());
        output.writeByte(w);
        output.writeByte(h);
        int lineCount = 2;
        try {
            do {
                for (int i = 0; i < h; i++) {
                    lineCount++;
                    String line = reader.readLine();
                    if (line == null) {
                        throw new RuntimeException("Unexpected end of file at line " + lineCount);
                    }
                    for (int j = 0; j < w; j++) {
                        char c = line.charAt(j);
                        System.out.print(c);
                        output.writeByte(c);
                    }
                    System.out.println("");
                }
                lineCount++;
                output.writeShort(Short.parseShort(reader.readLine()));
            } while (reader.readLine() != null);
        } finally {
            reader.close();
            output.close();
        }
    }
```

**B**

```java
private static void createNonCompoundData(String dir, String type) {
        try {
            Set s = new HashSet();
            File nouns = new File(dir + "index." + type);
            FileInputStream fis = new FileInputStream(nouns);
            InputStreamReader reader = new InputStreamReader(fis);
            StringBuffer sb = new StringBuffer();
            int chr = reader.read();
            while (chr >= 0) {
                if (chr == '\n' || chr == '\r') {
                    String line = sb.toString();
                    if (line.length() > 0) {
                        if (line.charAt(0) != ' ') {
                            String[] spaceSplit = PerlHelp.split(line);
                            if (spaceSplit[0].indexOf('_') < 0) {
                                s.add(spaceSplit[0]);
                            }
                        }
                    }
                    sb.setLength(0);
                } else {
                    sb.append((char) chr);
                }
                chr = reader.read();
            }
            System.out.println(type + " size=" + s.size());
            File output = new File(dir + "nonCompound." + type + "s.gz");
            FileOutputStream fos = new FileOutputStream(output);
            GZIPOutputStream gzos = new GZIPOutputStream(new BufferedOutputStream(fos));
            PrintWriter writer = new PrintWriter(gzos);
            writer.println("# This file was extracted from WordNet data, the following copyright notice");
            writer.println("# from WordNet is attached.");
            writer.println("#");
            writer.println("#  This software and database is being provided to you, the LICENSEE, by  ");
            writer.println("#  Princeton University under the following license.  By obtaining, using  ");
            writer.println("#  and/or copying this software and database, you agree that you have  ");
            writer.println("#  read, understood, and will comply with these terms and conditions.:  ");
            writer.println("#  ");
            writer.println("#  Permission to use, copy, modify and distribute this software and  ");
            writer.println("#  database and its documentation for any purpose and without fee or  ");
            writer.println("#  royalty is hereby granted, provided that you agree to comply with  ");
            writer.println("#  the following copyright notice and statements, including the disclaimer,  ");
            writer.println("#  and that the same appear on ALL copies of the software, database and  ");
            writer.println("#  documentation, including modifications that you make for internal  ");
            writer.println("#  use or for distribution.  ");
            writer.println("#  ");
            writer.println("#  WordNet 1.7 Copyright 2001 by Princeton University.  All rights reserved. ");
            writer.println("#  ");
            writer.println("#  THIS SOFTWARE AND DATABASE IS PROVIDED \"AS IS\" AND PRINCETON  ");
            writer.println("#  UNIVERSITY MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR  ");
            writer.println("#  IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, PRINCETON  ");
            writer.println("#  UNIVERSITY MAKES NO REPRESENTATIONS OR WARRANTIES OF MERCHANT-  ");
            writer.println("#  ABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE  ");
            writer.println("#  OF THE LICENSED SOFTWARE, DATABASE OR DOCUMENTATION WILL NOT  ");
            writer.println("#  INFRINGE ANY THIRD PARTY PATENTS, COPYRIGHTS, TRADEMARKS OR ");
            writer.println("#  OTHER RIGHTS. ");
            writer.println("#  ");
            writer.println("#  The name of Princeton University or Princeton may not be used in");
            writer.println("#  advertising or publicity pertaining to distribution of the software");
            writer.println("#  and/or database.  Title to copyright in this software, database and");
            writer.println("#  any associated documentation shall at all times remain with");
            writer.println("#  Princeton University and LICENSEE agrees to preserve same.  ");
            for (Iterator i = s.iterator(); i.hasNext(); ) {
                String mwe = (String) i.next();
                writer.println(mwe);
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
```

## P008

**A**

```java
public Blowfish(String password) {
        MessageDigest digest = null;
        try {
            digest = MessageDigest.getInstance("SHA1");
            digest.update(password.getBytes());
        } catch (Exception e) {
            Log.error(e.getMessage(), e);
        }
        m_bfish = new BlowfishCBC(digest.digest(), 0);
        digest.reset();
    }
```

**B**

```java
public String md5Encode(String pass) {
        MessageDigest md = null;
        try {
            md = MessageDigest.getInstance("MD5");
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        }
        md.update(pass.getBytes());
        byte[] result = md.digest();
        return new String(result);
    }
```

## P009

**A**

```java
@SuppressWarnings("finally")
  private void compress(File src) throws IOException
  {
    if (this.switches.contains(Switch.test))
      return;

    checkSourceFile(src);
    if (src.getPath().endsWith(".bz2"))
    {
      this.log.println("WARNING: skipping file because it already has .bz2 suffix:").println(src);
      return;
    }

    final File dst = new File(src.getPath() + ".bz2").getAbsoluteFile();
    if (!checkDestFile(dst))
      return;

    FileChannel       inChannel   = null;
    FileChannel       outChannel  = null;
    FileOutputStream  fileOut     = null;
    BZip2OutputStream bzOut       = null;
    FileLock          inLock      = null;
    FileLock          outLock     = null;

    try
    {
      inChannel = new FileInputStream(src).getChannel();
      final long inSize = inChannel.size();
      inLock = inChannel.tryLock(0, inSize, true);
      if (inLock == null)
        throw error("source file locked by another process: " + src);

      fileOut     = new FileOutputStream(dst);
      outChannel  = fileOut.getChannel();
      bzOut       = new BZip2OutputStream(
        new BufferedXOutputStream(fileOut, 8192),
        Math.min(
          (this.blockSize == -1) ? BZip2OutputStream.MAX_BLOCK_SIZE : this.blockSize,
          BZip2OutputStream.chooseBlockSize(inSize)
        )
      );

      outLock = outChannel.tryLock();
      if (outLock == null)
        throw error("destination file locked by another process: " + dst);

      final boolean showProgress = this.switches.contains(Switch.showProgress);
      long pos = 0;
      int progress = 0;

      if (showProgress || this.verbose)
      {
        this.log.print("source: " + src).print(": size=").println(inSize);
        this.log.println("target: " + dst);
      }

      while (true)
      {
        final long maxStep = showProgress ? Math.max(8192, (inSize - pos) / MAX_PROGRESS) : (inSize - pos);
        if (maxStep <= 0)
        {
          if (showProgress)
          {
            for (int i = progress; i < MAX_PROGRESS; i++)
              this.log.print('#');
            this.log.println(" done");
          }
          break;
        }
        else
        {
          final long step = inChannel.transferTo(pos, maxStep, bzOut);
          if ((step == 0) && (inChannel.size() != inSize))
            throw error("file " + src + " has been modified concurrently by another process");

          pos += step;
          if (showProgress)
          {
            final double  p           = (double) pos / (double) inSize;
            final int     newProgress = (int) (MAX_PROGRESS * p);
            for (int i = progress; i < newProgress; i++)
              this.log.print('#');
            progress = newProgress;
          }
        }
      }

      inLock.release();
      inChannel.close();
      bzOut.closeInstance();
      final long outSize = outChannel.position();
      outChannel.truncate(outSize);
      outLock.release();
      fileOut.close();

      if (this.verbose)
      {
        final double ratio = (inSize == 0) ? (outSize * 100) : ((double) outSize / (double) inSize);
        this.log.print("raw size: ").print(inSize)
          .print("; compressed size: ").print(outSize)
          .print("; compression ratio: ").print(ratio).println('%');
      }

      if (!this.switches.contains(Switch.keep))
      {
        if (!src.delete())
          throw error("unable to delete sourcefile: " + src);
      }
    }
    catch (final IOException ex)
    {
      IO.tryClose(inChannel);
      IO.tryClose(bzOut);
      IO.tryClose(fileOut);
      IO.tryRelease(inLock);
      IO.tryRelease(outLock);
      try
      {
        this.log.println();
      }
      finally
      {
        throw ex;
      }
    }
  }
```

**B**

```java
private final boolean copy_to_file_nio(File src, File dst) throws IOException {
        FileChannel srcChannel = null, dstChannel = null;
        try {
            srcChannel = new FileInputStream(src).getChannel();
            dstChannel = new FileOutputStream(dst).getChannel();
            {
                int safe_max = (64 * 1024 * 1024) / 4;
                long size = srcChannel.size();
                long position = 0;
                while (position < size) {
                    position += srcChannel.transferTo(position, safe_max, dstChannel);
                }
            }
            return true;
        } finally {
            try {
                if (srcChannel != null) srcChannel.close();
            } catch (IOException e) {
                Debug.debug(e);
            }
            try {
                if (dstChannel != null) dstChannel.close();
            } catch (IOException e) {
                Debug.debug(e);
            }
        }
    }
```

## P010

**A**

```java
public void writeFile(OutputStream outputStream) throws IOException {
            InputStream inputStream = null;
            if (file != null) {
                try {
                    inputStream = new FileInputStream(file);
                    IOUtils.copy(inputStream, outputStream);
                } finally {
                    if (inputStream != null) {
                        IOUtils.closeQuietly(inputStream);
                    }
                }
            }
        }
```

**B**

```java
private static void copyFile(File sourceFile, File destFile) {
        try {
            if (!destFile.exists()) {
                destFile.createNewFile();
            }
            FileChannel source = null;
            FileChannel destination = null;
            try {
                source = new FileInputStream(sourceFile).getChannel();
                destination = new FileOutputStream(destFile).getChannel();
                destination.transferFrom(source, 0, source.size());
            } finally {
                if (source != null) {
                    source.close();
                }
                if (destination != null) {
                    destination.close();
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
```

## P011

**A**

```java
private boolean passwordMatches(String user, String plainPassword, String scrambledPassword) {
        MessageDigest md;
        byte[] temp_digest, pass_digest;
        byte[] hex_digest = new byte[35];
        byte[] scrambled = scrambledPassword.getBytes();
        try {
            md = MessageDigest.getInstance("MD5");
            md.update(plainPassword.getBytes("US-ASCII"));
            md.update(user.getBytes("US-ASCII"));
            temp_digest = md.digest();
            Utils.bytesToHex(temp_digest, hex_digest, 0);
            md.update(hex_digest, 0, 32);
            md.update(salt.getBytes());
            pass_digest = md.digest();
            Utils.bytesToHex(pass_digest, hex_digest, 3);
            hex_digest[0] = (byte) 'm';
            hex_digest[1] = (byte) 'd';
            hex_digest[2] = (byte) '5';
            for (int i = 0; i < hex_digest.length; i++) {
                if (scrambled[i] != hex_digest[i]) {
                    return false;
                }
            }
        } catch (Exception e) {
            logger.error(e);
        }
        return true;
    }
```

**B**

```java
public String digest(String password, String digestType, String inputEncoding) throws CmsPasswordEncryptionException {
        MessageDigest md;
        String result;
        try {
            if (DIGEST_TYPE_PLAIN.equals(digestType.toLowerCase())) {
                result = password;
            } else if (DIGEST_TYPE_SSHA.equals(digestType.toLowerCase())) {
                byte[] salt = new byte[4];
                byte[] digest;
                byte[] total;
                if (m_secureRandom == null) {
                    m_secureRandom = SecureRandom.getInstance("SHA1PRNG");
                }
                m_secureRandom.nextBytes(salt);
                md = MessageDigest.getInstance(DIGEST_TYPE_SHA);
                md.reset();
                md.update(password.getBytes(inputEncoding));
                md.update(salt);
                digest = md.digest();
                total = new byte[digest.length + salt.length];
                System.arraycopy(digest, 0, total, 0, digest.length);
                System.arraycopy(salt, 0, total, digest.length, salt.length);
                result = new String(Base64.encodeBase64(total));
            } else {
                md = MessageDigest.getInstance(digestType);
                md.reset();
                md.update(password.getBytes(inputEncoding));
                result = new String(Base64.encodeBase64(md.digest()));
            }
        } catch (NoSuchAlgorithmException e) {
            CmsMessageContainer message = Messages.get().container(Messages.ERR_UNSUPPORTED_ALGORITHM_1, digestType);
            if (LOG.isErrorEnabled()) {
                LOG.error(message.key(), e);
            }
            throw new CmsPasswordEncryptionException(message, e);
        } catch (UnsupportedEncodingException e) {
            CmsMessageContainer message = Messages.get().container(Messages.ERR_UNSUPPORTED_PASSWORD_ENCODING_1, inputEncoding);
            if (LOG.isErrorEnabled()) {
                LOG.error(message.key(), e);
            }
            throw new CmsPasswordEncryptionException(message, e);
        }
        return result;
    }
```

## P012

**A**

```java
public static void main(String[] args) throws IOException {
        String uri = "hdfs://localhost:8020/user/leeing/maxtemp/sample.txt";
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(URI.create(uri), conf);
        FSDataInputStream in = null;
        try {
            in = fs.open(new Path(uri));
            IOUtils.copyBytes(in, System.out, 8192, false);
            System.out.println("\n");
            in.seek(0);
            IOUtils.copyBytes(in, System.out, 8192, false);
        } finally {
            IOUtils.closeStream(in);
        }
    }
```

**B**

```java
public static void main(String[] args) throws Exception {
        String codecClassname = args[0];
        Class<?> codecClass = Class.forName(codecClassname);
        Configuration conf = new Configuration();
        CompressionCodec codec = (CompressionCodec) ReflectionUtils.newInstance(codecClass, conf);
        Compressor compressor = null;
        try {
            compressor = CodecPool.getCompressor(codec);
            CompressionOutputStream out = codec.createOutputStream(System.out, compressor);
            IOUtils.copyBytes(System.in, out, 4096, false);
            out.finish();
        } finally {
            CodecPool.returnCompressor(compressor);
        }
    }
```

## P013

**A**

```java
public boolean refresh() {
        try {
            synchronized (text) {
                stream = (new URL(url)).openStream();
                BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
                String line;
                StringBuilder sb = new StringBuilder();
                while ((line = reader.readLine()) != null) {
                    sb.append(line);
                    sb.append("\n");
                }
                text = sb.toString();
            }
            price = 0;
            date = null;
        } catch (MalformedURLException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        } finally {
            if (stream != null) try {
                stream.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return true;
    }
```

**B**

```java
private void getRdfResponse(StringBuilder sb, String url) {
        try {
            String inputLine = null;
            BufferedReader reader = new BufferedReader(new InputStreamReader(new URL(url).openStream()));
            while ((inputLine = reader.readLine()) != null) {
                sb.append(inputLine);
            }
            reader.close();
        } catch (MalformedURLException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
```

## P014

**A**

```java
private static void main(String mp3Path) throws IOException {
        String convPath = "http://android.adinterest.biz/wav2mp3.php?k=";
        String uri = convPath + mp3Path;
        URL rssurl = new URL(uri);
        InputStream is = rssurl.openStream();
        BufferedReader br = new BufferedReader(new InputStreamReader(is, "UTF-8"));
        String buf = "";
        while ((buf = br.readLine()) != null) {
        }
        is.close();
        br.close();
    }
```

**B**

```java
private File copyFromURL(URL url, String dir) throws IOException {
        File urlFile = new File(url.getFile());
        File dest = new File(dir, urlFile.getName());
        logger.log("Extracting " + urlFile.getName() + " to " + dir + "...");
        FileOutputStream os = new FileOutputStream(dest);
        InputStream is = url.openStream();
        byte data[] = new byte[4096];
        int ct;
        while ((ct = is.read(data)) >= 0) os.write(data, 0, ct);
        is.close();
        os.close();
        logger.log("ok\n");
        return dest;
    }
```

## P015

**A**

```java
private String convert(InputStream input, String encoding) throws Exception {
        Process p = Runtime.getRuntime().exec("tidy -q -f /dev/null -wrap 0 --output-xml yes --doctype omit --force-output true --new-empty-tags  " + emptyTags + " --quote-nbsp no -utf8");
        Thread t = new CopyThread(input, p.getOutputStream());
        t.start();
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        IOUtils.copy(p.getInputStream(), output);
        p.waitFor();
        t.join();
        return output.toString();
    }
```

**B**

```java
public void testWriteThreadsNoCompression() throws Exception {
        Bootstrap bootstrap = new Bootstrap();
        bootstrap.loadProfiles(CommandLineProcessorFactory.PROFILE.DB, CommandLineProcessorFactory.PROFILE.REST_CLIENT, CommandLineProcessorFactory.PROFILE.COLLECTOR);
        final LocalLogFileWriter writer = (LocalLogFileWriter) bootstrap.getBean(LogFileWriter.class);
        writer.init();
        writer.setCompressionCodec(null);
        File fileInput = new File(baseDir, "testWriteOneFile/input");
        fileInput.mkdirs();
        File fileOutput = new File(baseDir, "testWriteOneFile/output");
        fileOutput.mkdirs();
        writer.setBaseDir(fileOutput);
        int fileCount = 100;
        int lineCount = 100;
        File[] inputFiles = createInput(fileInput, fileCount, lineCount);
        ExecutorService exec = Executors.newFixedThreadPool(fileCount);
        final CountDownLatch latch = new CountDownLatch(fileCount);
        for (int i = 0; i < fileCount; i++) {
            final File file = inputFiles[i];
            final int count = i;
            exec.submit(new Callable<Boolean>() {

                @Override
                public Boolean call() throws Exception {
                    FileStatus.FileTrackingStatus status = FileStatus.FileTrackingStatus.newBuilder().setFileDate(System.currentTimeMillis()).setDate(System.currentTimeMillis()).setAgentName("agent1").setFileName(file.getName()).setFileSize(file.length()).setLogType("type1").build();
                    BufferedReader reader = new BufferedReader(new FileReader(file));
                    try {
                        String line = null;
                        while ((line = reader.readLine()) != null) {
                            writer.write(status, new ByteArrayInputStream((line + "\n").getBytes()));
                        }
                    } finally {
                        IOUtils.closeQuietly(reader);
                    }
                    LOG.info("Thread[" + count + "] completed ");
                    latch.countDown();
                    return true;
                }
            });
        }
        latch.await();
        exec.shutdown();
        LOG.info("Shutdown thread service");
        writer.close();
        File[] outputFiles = fileOutput.listFiles();
        assertNotNull(outputFiles);
        File testCombinedInput = new File(baseDir, "combinedInfile.txt");
        testCombinedInput.createNewFile();
        FileOutputStream testCombinedInputOutStream = new FileOutputStream(testCombinedInput);
        try {
            for (File file : inputFiles) {
                FileInputStream f1In = new FileInputStream(file);
                IOUtils.copy(f1In, testCombinedInputOutStream);
            }
        } finally {
            testCombinedInputOutStream.close();
        }
        File testCombinedOutput = new File(baseDir, "combinedOutfile.txt");
        testCombinedOutput.createNewFile();
        FileOutputStream testCombinedOutOutStream = new FileOutputStream(testCombinedOutput);
        try {
            System.out.println("----------------- " + testCombinedOutput.getAbsolutePath());
            for (File file : outputFiles) {
                FileInputStream f1In = new FileInputStream(file);
                IOUtils.copy(f1In, testCombinedOutOutStream);
            }
        } finally {
            testCombinedOutOutStream.close();
        }
        FileUtils.contentEquals(testCombinedInput, testCombinedOutput);
    }
```

## P016

**A**

```java
private void handleSSI(HttpData data) throws HttpError, IOException {
        File tempFile = TempFileHandler.getTempFile();
        FileOutputStream out = new FileOutputStream(tempFile);
        BufferedReader in = new BufferedReader(new FileReader(data.realPath));
        String[] env = getEnvironmentVariables(data);
        if (ssi == null) {
            ssi = new BSssi();
        }
        ssi.addEnvironment(env);
        if (data.resp == null) {
            SimpleResponse resp = new SimpleResponse();
            resp.setHeader("Content-Type", "text/html");
            moreHeaders(resp);
            resp.setHeader("Connection", "close");
            data.resp = resp;
            resp.write(data.out);
        }
        String t;
        int start;
        Enumeration en;
        boolean anIfCondition = true;
        while ((t = in.readLine()) != null) {
            if ((start = t.indexOf("<!--#")) > -1) {
                if (anIfCondition) out.write(t.substring(0, start).getBytes());
                try {
                    en = ssi.parse(t.substring(start)).elements();
                    SSICommand command;
                    while (en.hasMoreElements()) {
                        command = (SSICommand) en.nextElement();
                        logger.fine("Command=" + command);
                        switch(command.getCommand()) {
                            case BSssi.CMD_IF_TRUE:
                                anIfCondition = true;
                                break;
                            case BSssi.CMD_IF_FALSE:
                                anIfCondition = false;
                                break;
                            case BSssi.CMD_CGI:
                                out.flush();
                                if (command.getFileType() != null && command.getFileType().startsWith("shtm")) {
                                    HttpData d = newHttpData(data);
                                    d.out = out;
                                    d.realPath = HttpThread.getMappedFilename(command.getMessage(), data.req.getUrl());
                                    new SsiHandler(d, ssi).perform();
                                } else {
                                    String application = getExtension(command.getFileType());
                                    if (application == null) {
                                        writePaused(new FileInputStream(HttpThread.getMappedFilename(command.getMessage(), data.req.getUrl())), out, pause);
                                    } else {
                                        String parameter = "";
                                        if (command.getMessage().indexOf("php") >= 0) {
                                            parameter = "-f ";
                                        }
                                        Process p = Runtime.getRuntime().exec(application + " " + parameter + HttpThread.getMappedFilename(command.getMessage(), data.req.getUrl()));
                                        BufferedReader pIn = new BufferedReader(new InputStreamReader(p.getInputStream()));
                                        String aLine;
                                        while ((aLine = pIn.readLine()) != null) out.write((aLine + "\n").getBytes());
                                        pIn.close();
                                    }
                                }
                                break;
                            case BSssi.CMD_EXEC:
                                Process p = Runtime.getRuntime().exec(command.getMessage());
                                BufferedReader pIn = new BufferedReader(new InputStreamReader(p.getInputStream()));
                                String aLine;
                                while ((aLine = pIn.readLine()) != null) out.write((aLine + "\n").getBytes());
                                BufferedReader pErr = new BufferedReader(new InputStreamReader(p.getErrorStream()));
                                while ((aLine = pErr.readLine()) != null) out.write((aLine + "\n").getBytes());
                                pIn.close();
                                pErr.close();
                                p.destroy();
                                break;
                            case BSssi.CMD_INCLUDE:
                                File incFile = HttpThread.getMappedFilename(command.getMessage());
                                if (incFile.exists() && incFile.canRead()) {
                                    writePaused(new FileInputStream(incFile), out, pause);
                                }
                                break;
                            case BSssi.CMD_FILESIZE:
                                long sizeBytes = HttpThread.getMappedFilename(command.getMessage(), data.req.getUrl()).length();
                                double smartSize;
                                String unit = "bytes";
                                if (command.getFileType().trim().equals("abbrev")) {
                                    if (sizeBytes > 1000000) {
                                        smartSize = sizeBytes / 1024000.0;
                                        unit = "M";
                                    } else if (sizeBytes > 1000) {
                                        smartSize = sizeBytes / 1024.0;
                                        unit = "K";
                                    } else {
                                        smartSize = sizeBytes;
                                        unit = "bytes";
                                    }
                                    NumberFormat numberFormat = new DecimalFormat("#,##0", new DecimalFormatSymbols(Locale.ENGLISH));
                                    out.write((numberFormat.format(smartSize) + "" + unit).getBytes());
                                } else {
                                    NumberFormat numberFormat = new DecimalFormat("#,###,##0", new DecimalFormatSymbols(Locale.ENGLISH));
                                    out.write((numberFormat.format(sizeBytes) + " " + unit).getBytes());
                                }
                                break;
                            case BSssi.CMD_FLASTMOD:
                                out.write(ssi.format(new Date(HttpThread.getMappedFilename(command.getMessage(), data.req.getUrl()).lastModified()), TimeZone.getTimeZone("GMT")).getBytes());
                                break;
                            case BSssi.CMD_NOECHO:
                                break;
                            case BSssi.CMD_ECHO:
                            default:
                                out.write(command.getMessage().getBytes());
                                break;
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                    out.write((ssi.getErrorMessage() + " " + e.getMessage()).getBytes());
                }
                if (anIfCondition) out.write("\n".getBytes());
            } else {
                if (anIfCondition) out.write((t + "\n").getBytes());
            }
            out.flush();
        }
        in.close();
        out.close();
        data.fileData.setContentType("text/html");
        data.fileData.setFile(tempFile);
        writePaused(new FileInputStream(tempFile), data.out, pause);
        logger.fine("HandleSSI done for " + data.resp);
    }
```

**B**

```java
public String getHttpText() {
        URL url = null;
        try {
            url = new URL(getUrl());
        } catch (MalformedURLException e) {
            log.error(e.getMessage());
        }
        StringBuffer sb = new StringBuffer();
        HttpURLConnection conn = null;
        try {
            conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod(getRequestMethod());
            conn.setDoOutput(true);
            if (getRequestProperty() != null && "".equals(getRequestProperty())) {
                conn.setRequestProperty("Accept", getRequestProperty());
            }
            PrintWriter out = new PrintWriter(new OutputStreamWriter(conn.getOutputStream(), getCharset()));
            out.println(getParam());
            out.close();
            BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream(), getCharset()));
            String inputLine;
            int i = 1;
            while ((inputLine = in.readLine()) != null) {
                if (getStartLine() == 0 && getEndLine() == 0) {
                    sb.append(inputLine).append("\n");
                } else {
                    if (getEndLine() > 0) {
                        if (i >= getStartLine() && i <= getEndLine()) {
                            sb.append(inputLine).append("\n");
                        }
                    } else {
                        if (i >= getStartLine()) {
                            sb.append(inputLine).append("\n");
                        }
                    }
                }
                i++;
            }
            in.close();
        } catch (IOException e) {
            log.error(e.getMessage());
        } finally {
            if (conn != null) {
                conn.disconnect();
            }
        }
        return sb.toString();
    }
```

## P017

**A**

```java
@Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        String contentId = req.getParameter(CONTENT_ID);
        String contentType = req.getParameter(CONTENT_TYPE);
        if (contentId == null || contentType == null) {
            resp.sendError(HttpServletResponse.SC_BAD_REQUEST, "Content id or content type not specified");
            return;
        }
        try {
            switch(ContentType.valueOf(contentType)) {
                case IMAGE:
                    resp.setContentType("image/jpeg");
                    break;
                case AUDIO:
                    resp.setContentType("audio/mp3");
                    break;
                case VIDEO:
                    resp.setContentType("video/mpeg");
                    break;
                default:
                    throw new IllegalStateException("Invalid content type specified");
            }
        } catch (IllegalArgumentException e) {
            resp.sendError(HttpServletResponse.SC_BAD_REQUEST, "Invalid content type specified");
            return;
        }
        String baseUrl = this.getServletContext().getInitParameter(BASE_URL);
        URL url = new URL(baseUrl + "/" + contentType.toLowerCase() + "/" + contentId);
        URLConnection conn = url.openConnection();
        resp.setContentLength(conn.getContentLength());
        IOUtils.copy(conn.getInputStream(), resp.getOutputStream());
    }
```

**B**

```java
@Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        String uuid = req.getParameterValues(Constants.PARAM_UUID)[0];
        String datastream = null;
        if (req.getRequestURI().contains(Constants.SERVLET_DOWNLOAD_FOXML_PREFIX)) {
            resp.addHeader("Content-Disposition", "attachment; ContentType = \"text/xml\"; filename=\"" + uuid + "_server_version.foxml\"");
        } else {
            datastream = req.getParameterValues(Constants.PARAM_DATASTREAM)[0];
            resp.addHeader("Content-Disposition", "attachment; ContentType = \"text/xml\"; filename=\"" + uuid + "_server_version_" + datastream + ".xml\"");
        }
        ServletOutputStream os = resp.getOutputStream();
        if (uuid != null && !"".equals(uuid)) {
            try {
                StringBuffer sb = new StringBuffer();
                if (req.getRequestURI().contains(Constants.SERVLET_DOWNLOAD_FOXML_PREFIX)) {
                    sb.append(config.getFedoraHost()).append("/objects/").append(uuid).append("/objectXML");
                } else if (req.getRequestURI().contains(Constants.SERVLET_DOWNLOAD_DATASTREAMS_PREFIX)) {
                    sb.append(config.getFedoraHost()).append("/objects/").append(uuid).append("/datastreams/").append(datastream).append("/content");
                }
                InputStream is = RESTHelper.get(sb.toString(), config.getFedoraLogin(), config.getFedoraPassword(), false);
                if (is == null) {
                    return;
                }
                try {
                    if (req.getRequestURI().contains(Constants.SERVLET_DOWNLOAD_DATASTREAMS_PREFIX)) {
                        os.write(Constants.XML_HEADER_WITH_BACKSLASHES.getBytes());
                    }
                    IOUtils.copyStreams(is, os);
                } catch (IOException e) {
                    resp.setStatus(HttpURLConnection.HTTP_NOT_FOUND);
                    LOGGER.error("Problem with downloading foxml.", e);
                } finally {
                    os.flush();
                    if (is != null) {
                        try {
                            is.close();
                        } catch (IOException e) {
                            resp.setStatus(HttpURLConnection.HTTP_NOT_FOUND);
                            LOGGER.error("Problem with downloading foxml.", e);
                        } finally {
                            is = null;
                        }
                    }
                }
            } catch (IOException e) {
                resp.setStatus(HttpURLConnection.HTTP_NOT_FOUND);
                LOGGER.error("Problem with downloading foxml.", e);
            } finally {
                os.flush();
            }
        }
    }
```

## P018

**A**

```java
private static void doCopyFile(File srcFile, File destFile, boolean preserveFileDate) throws IOException {
        if (destFile.exists() && destFile.isDirectory()) {
            throw new IOException("Destination '" + destFile + "' exists but is a directory");
        }
        FileInputStream input = new FileInputStream(srcFile);
        try {
            FileOutputStream output = new FileOutputStream(destFile);
            try {
                IOUtils.copy(input, output);
            } finally {
                IOUtils.closeQuietly(output);
            }
        } finally {
            IOUtils.closeQuietly(input);
        }
        if (srcFile.length() != destFile.length()) {
            throw new IOException("Failed to copy full contents from '" + srcFile + "' to '" + destFile + "'");
        }
        if (preserveFileDate) {
            destFile.setLastModified(srcFile.lastModified());
        }
    }
```

**B**

```java
protected void copyFile(File sourceFile, File destFile) {
        FileChannel in = null;
        FileChannel out = null;
        try {
            if (!verifyOrCreateParentPath(destFile.getParentFile())) {
                throw new IOException("Parent directory path " + destFile.getAbsolutePath() + " did not exist and could not be created");
            }
            if (destFile.exists() || destFile.createNewFile()) {
                in = new FileInputStream(sourceFile).getChannel();
                out = new FileOutputStream(destFile).getChannel();
                in.transferTo(0, in.size(), out);
            } else {
                throw new IOException("Couldn't create file for " + destFile.getAbsolutePath());
            }
        } catch (IOException ioe) {
            if (destFile.exists() && destFile.length() < sourceFile.length()) {
                destFile.delete();
            }
            ioe.printStackTrace();
        } finally {
            try {
                in.close();
            } catch (Throwable t) {
            }
            try {
                out.close();
            } catch (Throwable t) {
            }
            destFile.setLastModified(sourceFile.lastModified());
        }
    }
```

## P019

**A**

```java
public void launchJob(final String workingDir, final AppConfigType appConfig) throws FaultType {
        logger.info("called for job: " + jobID);
        MessageContext mc = MessageContext.getCurrentContext();
        HttpServletRequest req = (HttpServletRequest) mc.getProperty(HTTPConstants.MC_HTTP_SERVLETREQUEST);
        String clientDN = (String) req.getAttribute(GSIConstants.GSI_USER_DN);
        if (clientDN != null) {
            logger.info("Client's DN: " + clientDN);
        } else {
            clientDN = "Unknown client";
        }
        String remoteIP = req.getRemoteAddr();
        SOAPService service = mc.getService();
        String serviceName = service.getName();
        if (serviceName == null) {
            serviceName = "Unknown service";
        }
        if (appConfig.isParallel()) {
            if (AppServiceImpl.drmaaInUse) {
                if (AppServiceImpl.drmaaPE == null) {
                    logger.error("drmaa.pe property must be specified in opal.properties " + "for parallel execution using DRMAA");
                    throw new FaultType("drmaa.pe property must be specified in opal.properties " + "for parallel execution using DRMAA");
                }
                if (AppServiceImpl.mpiRun == null) {
                    logger.error("mpi.run property must be specified in opal.properties " + "for parallel execution using DRMAA");
                    throw new FaultType("mpi.run property must be specified in " + "opal.properties for parallel execution " + "using DRMAA");
                }
            } else if (!AppServiceImpl.globusInUse) {
                if (AppServiceImpl.mpiRun == null) {
                    logger.error("mpi.run property must be specified in opal.properties " + "for parallel execution without using Globus");
                    throw new FaultType("mpi.run property must be specified in " + "opal.properties for parallel execution " + "without using Globus");
                }
            }
            if (jobIn.getNumProcs() == null) {
                logger.error("Number of processes unspecified for parallel job");
                throw new FaultType("Number of processes unspecified for parallel job");
            } else if (jobIn.getNumProcs().intValue() > AppServiceImpl.numProcs) {
                logger.error("Processors required - " + jobIn.getNumProcs() + ", available - " + AppServiceImpl.numProcs);
                throw new FaultType("Processors required - " + jobIn.getNumProcs() + ", available - " + AppServiceImpl.numProcs);
            }
        }
        try {
            status.setCode(GramJob.STATUS_PENDING);
            status.setMessage("Launching executable");
            status.setBaseURL(new URI(AppServiceImpl.tomcatURL + jobID));
        } catch (MalformedURIException mue) {
            logger.error("Cannot convert base_url string to URI - " + mue.getMessage());
            throw new FaultType("Cannot convert base_url string to URI - " + mue.getMessage());
        }
        if (!AppServiceImpl.dbInUse) {
            AppServiceImpl.statusTable.put(jobID, status);
        } else {
            Connection conn = null;
            try {
                conn = DriverManager.getConnection(AppServiceImpl.dbUrl, AppServiceImpl.dbUser, AppServiceImpl.dbPasswd);
            } catch (SQLException e) {
                logger.error("Cannot connect to database - " + e.getMessage());
                throw new FaultType("Cannot connect to database - " + e.getMessage());
            }
            String time = new SimpleDateFormat("MMM d, yyyy h:mm:ss a", Locale.US).format(new Date());
            String sqlStmt = "insert into job_status(job_id, code, message, base_url, " + "client_dn, client_ip, service_name, start_time, last_update) " + "values ('" + jobID + "', " + status.getCode() + ", " + "'" + status.getMessage() + "', " + "'" + status.getBaseURL() + "', " + "'" + clientDN + "', " + "'" + remoteIP + "', " + "'" + serviceName + "', " + "'" + time + "', " + "'" + time + "');";
            try {
                Statement stmt = conn.createStatement();
                stmt.executeUpdate(sqlStmt);
                conn.close();
            } catch (SQLException e) {
                logger.error("Cannot insert job status into database - " + e.getMessage());
                throw new FaultType("Cannot insert job status into database - " + e.getMessage());
            }
        }
        String args = appConfig.getDefaultArgs();
        if (args == null) {
            args = jobIn.getArgList();
        } else {
            String userArgs = jobIn.getArgList();
            if (userArgs != null) args += " " + userArgs;
        }
        if (args != null) {
            args = args.trim();
        }
        logger.debug("Argument list: " + args);
        if (AppServiceImpl.drmaaInUse) {
            String cmd = null;
            String[] argsArray = null;
            if (appConfig.isParallel()) {
                cmd = "/bin/sh";
                String newArgs = AppServiceImpl.mpiRun + " -machinefile $TMPDIR/machines" + " -np " + jobIn.getNumProcs() + " " + appConfig.getBinaryLocation();
                if (args != null) {
                    args = newArgs + " " + args;
                } else {
                    args = newArgs;
                }
                logger.debug("CMD: " + args);
                argsArray = new String[] { "-c", args };
            } else {
                cmd = appConfig.getBinaryLocation();
                if (args == null) args = "";
                logger.debug("CMD: " + cmd + " " + args);
                argsArray = args.split(" ");
            }
            try {
                logger.debug("Working directory: " + workingDir);
                JobTemplate jt = session.createJobTemplate();
                if (appConfig.isParallel()) jt.setNativeSpecification("-pe " + AppServiceImpl.drmaaPE + " " + jobIn.getNumProcs());
                jt.setRemoteCommand(cmd);
                jt.setArgs(argsArray);
                jt.setJobName(jobID);
                jt.setWorkingDirectory(workingDir);
                jt.setErrorPath(":" + workingDir + "/stderr.txt");
                jt.setOutputPath(":" + workingDir + "/stdout.txt");
                drmaaJobID = session.runJob(jt);
                logger.info("DRMAA job has been submitted with id " + drmaaJobID);
                session.deleteJobTemplate(jt);
            } catch (Exception ex) {
                logger.error(ex);
                status.setCode(GramJob.STATUS_FAILED);
                status.setMessage("Error while running executable via DRMAA - " + ex.getMessage());
                if (AppServiceImpl.dbInUse) {
                    try {
                        updateStatusInDatabase(jobID, status);
                    } catch (SQLException e) {
                        logger.error(e);
                        throw new FaultType("Cannot update status into database - " + e.getMessage());
                    }
                }
                return;
            }
            status.setCode(GramJob.STATUS_ACTIVE);
            status.setMessage("Execution in progress");
            if (AppServiceImpl.dbInUse) {
                try {
                    updateStatusInDatabase(jobID, status);
                } catch (SQLException e) {
                    logger.error(e);
                    throw new FaultType("Cannot update status into database - " + e.getMessage());
                }
            }
        } else if (AppServiceImpl.globusInUse) {
            String rsl = null;
            if (appConfig.isParallel()) {
                rsl = "&(directory=" + workingDir + ")" + "(executable=" + appConfig.getBinaryLocation() + ")" + "(count=" + jobIn.getNumProcs() + ")" + "(jobtype=mpi)" + "(stdout=stdout.txt)" + "(stderr=stderr.txt)";
            } else {
                rsl = "&(directory=" + workingDir + ")" + "(executable=" + appConfig.getBinaryLocation() + ")" + "(stdout=stdout.txt)" + "(stderr=stderr.txt)";
            }
            if (args != null) {
                args = "\"" + args + "\"";
                args = args.replaceAll("[\\s]+", "\" \"");
                rsl += "(arguments=" + args + ")";
            }
            logger.debug("RSL: " + rsl);
            try {
                job = new GramJob(rsl);
                GlobusCredential globusCred = new GlobusCredential(AppServiceImpl.serviceCertPath, AppServiceImpl.serviceKeyPath);
                GSSCredential gssCred = new GlobusGSSCredentialImpl(globusCred, GSSCredential.INITIATE_AND_ACCEPT);
                job.setCredentials(gssCred);
                job.addListener(this);
                job.request(AppServiceImpl.gatekeeperContact);
            } catch (Exception ge) {
                logger.error(ge);
                status.setCode(GramJob.STATUS_FAILED);
                status.setMessage("Error while running executable via Globus - " + ge.getMessage());
                if (AppServiceImpl.dbInUse) {
                    try {
                        updateStatusInDatabase(jobID, status);
                    } catch (SQLException e) {
                        logger.error(e);
                        throw new FaultType("Cannot update status into database - " + e.getMessage());
                    }
                }
                return;
            }
        } else {
            String cmd = null;
            if (appConfig.isParallel()) {
                cmd = new String(AppServiceImpl.mpiRun + " " + "-np " + jobIn.getNumProcs() + " " + appConfig.getBinaryLocation());
            } else {
                cmd = new String(appConfig.getBinaryLocation());
            }
            if (args != null) {
                cmd += " " + args;
            }
            logger.debug("CMD: " + cmd);
            try {
                logger.debug("Working directory: " + workingDir);
                proc = Runtime.getRuntime().exec(cmd, null, new File(workingDir));
                stdoutThread = writeStdOut(proc, workingDir);
                stderrThread = writeStdErr(proc, workingDir);
            } catch (IOException ioe) {
                logger.error(ioe);
                status.setCode(GramJob.STATUS_FAILED);
                status.setMessage("Error while running executable via fork - " + ioe.getMessage());
                if (AppServiceImpl.dbInUse) {
                    try {
                        updateStatusInDatabase(jobID, status);
                    } catch (SQLException e) {
                        logger.error(e);
                        throw new FaultType("Cannot update status into database - " + e.getMessage());
                    }
                }
                return;
            }
            status.setCode(GramJob.STATUS_ACTIVE);
            status.setMessage("Execution in progress");
            if (AppServiceImpl.dbInUse) {
                try {
                    updateStatusInDatabase(jobID, status);
                } catch (SQLException e) {
                    logger.error(e);
                    throw new FaultType("Cannot update status into database - " + e.getMessage());
                }
            }
        }
        new Thread() {

            public void run() {
                try {
                    waitForCompletion();
                } catch (FaultType f) {
                    logger.error(f);
                    synchronized (status) {
                        status.notifyAll();
                    }
                    return;
                }
                if (AppServiceImpl.drmaaInUse || !AppServiceImpl.globusInUse) {
                    done = true;
                    status.setCode(GramJob.STATUS_STAGE_OUT);
                    status.setMessage("Writing output metadata");
                    if (AppServiceImpl.dbInUse) {
                        try {
                            updateStatusInDatabase(jobID, status);
                        } catch (SQLException e) {
                            status.setCode(GramJob.STATUS_FAILED);
                            status.setMessage("Cannot update status database after finish - " + e.getMessage());
                            logger.error(e);
                            synchronized (status) {
                                status.notifyAll();
                            }
                            return;
                        }
                    }
                }
                try {
                    if (!AppServiceImpl.drmaaInUse && !AppServiceImpl.globusInUse) {
                        try {
                            logger.debug("Waiting for all outputs to be written out");
                            stdoutThread.join();
                            stderrThread.join();
                            logger.debug("All outputs successfully written out");
                        } catch (InterruptedException ignore) {
                        }
                    }
                    File stdOutFile = new File(workingDir + File.separator + "stdout.txt");
                    if (!stdOutFile.exists()) {
                        throw new IOException("Standard output missing for execution");
                    }
                    File stdErrFile = new File(workingDir + File.separator + "stderr.txt");
                    if (!stdErrFile.exists()) {
                        throw new IOException("Standard error missing for execution");
                    }
                    if (AppServiceImpl.archiveData) {
                        logger.debug("Archiving output files");
                        File f = new File(workingDir);
                        File[] outputFiles = f.listFiles();
                        ZipOutputStream out = new ZipOutputStream(new FileOutputStream(workingDir + File.separator + jobID + ".zip"));
                        byte[] buf = new byte[1024];
                        try {
                            for (int i = 0; i < outputFiles.length; i++) {
                                FileInputStream in = new FileInputStream(outputFiles[i]);
                                out.putNextEntry(new ZipEntry(outputFiles[i].getName()));
                                int len;
                                while ((len = in.read(buf)) > 0) {
                                    out.write(buf, 0, len);
                                }
                                out.closeEntry();
                                in.close();
                            }
                            out.close();
                        } catch (IOException e) {
                            logger.error(e);
                            logger.error("Error not fatal - moving on");
                        }
                    }
                    File f = new File(workingDir);
                    File[] outputFiles = f.listFiles();
                    OutputFileType[] outputFileObj = new OutputFileType[outputFiles.length - 2];
                    int j = 0;
                    for (int i = 0; i < outputFiles.length; i++) {
                        if (outputFiles[i].getName().equals("stdout.txt")) {
                            outputs.setStdOut(new URI(AppServiceImpl.tomcatURL + jobID + "/stdout.txt"));
                        } else if (outputFiles[i].getName().equals("stderr.txt")) {
                            outputs.setStdErr(new URI(AppServiceImpl.tomcatURL + jobID + "/stderr.txt"));
                        } else {
                            OutputFileType next = new OutputFileType();
                            next.setName(outputFiles[i].getName());
                            next.setUrl(new URI(AppServiceImpl.tomcatURL + jobID + "/" + outputFiles[i].getName()));
                            outputFileObj[j++] = next;
                        }
                    }
                    outputs.setOutputFile(outputFileObj);
                } catch (IOException e) {
                    status.setCode(GramJob.STATUS_FAILED);
                    status.setMessage("Cannot retrieve outputs after finish - " + e.getMessage());
                    logger.error(e);
                    if (AppServiceImpl.dbInUse) {
                        try {
                            updateStatusInDatabase(jobID, status);
                        } catch (SQLException se) {
                            logger.error(se);
                        }
                    }
                    synchronized (status) {
                        status.notifyAll();
                    }
                    return;
                }
                if (!AppServiceImpl.dbInUse) {
                    AppServiceImpl.outputTable.put(jobID, outputs);
                } else {
                    Connection conn = null;
                    try {
                        conn = DriverManager.getConnection(AppServiceImpl.dbUrl, AppServiceImpl.dbUser, AppServiceImpl.dbPasswd);
                    } catch (SQLException e) {
                        status.setCode(GramJob.STATUS_FAILED);
                        status.setMessage("Cannot connect to database after finish - " + e.getMessage());
                        logger.error(e);
                        synchronized (status) {
                            status.notifyAll();
                        }
                        return;
                    }
                    String sqlStmt = "insert into job_output(job_id, std_out, std_err) " + "values ('" + jobID + "', " + "'" + outputs.getStdOut().toString() + "', " + "'" + outputs.getStdErr().toString() + "');";
                    Statement stmt = null;
                    try {
                        stmt = conn.createStatement();
                        stmt.executeUpdate(sqlStmt);
                    } catch (SQLException e) {
                        status.setCode(GramJob.STATUS_FAILED);
                        status.setMessage("Cannot update job output database after finish - " + e.getMessage());
                        logger.error(e);
                        try {
                            updateStatusInDatabase(jobID, status);
                        } catch (SQLException se) {
                            logger.error(se);
                        }
                        synchronized (status) {
                            status.notifyAll();
                        }
                        return;
                    }
                    OutputFileType[] outputFile = outputs.getOutputFile();
                    for (int i = 0; i < outputFile.length; i++) {
                        sqlStmt = "insert into output_file(job_id, name, url) " + "values ('" + jobID + "', " + "'" + outputFile[i].getName() + "', " + "'" + outputFile[i].getUrl().toString() + "');";
                        try {
                            stmt = conn.createStatement();
                            stmt.executeUpdate(sqlStmt);
                        } catch (SQLException e) {
                            status.setCode(GramJob.STATUS_FAILED);
                            status.setMessage("Cannot update output_file DB after finish - " + e.getMessage());
                            logger.error(e);
                            try {
                                updateStatusInDatabase(jobID, status);
                            } catch (SQLException se) {
                                logger.error(se);
                            }
                            synchronized (status) {
                                status.notifyAll();
                            }
                            return;
                        }
                    }
                }
                if (terminatedOK()) {
                    status.setCode(GramJob.STATUS_DONE);
                    status.setMessage("Execution complete - " + "check outputs to verify successful execution");
                } else {
                    status.setCode(GramJob.STATUS_FAILED);
                    status.setMessage("Execution failed");
                }
                if (AppServiceImpl.dbInUse) {
                    try {
                        updateStatusInDatabase(jobID, status);
                    } catch (SQLException e) {
                        status.setCode(GramJob.STATUS_FAILED);
                        status.setMessage("Cannot update status database after finish - " + e.getMessage());
                        logger.error(e);
                        synchronized (status) {
                            status.notifyAll();
                        }
                        return;
                    }
                }
                AppServiceImpl.jobTable.remove(jobID);
                synchronized (status) {
                    status.notifyAll();
                }
                logger.info("Execution complete for job: " + jobID);
            }
        }.start();
    }
```

**B**

```java
public void GetFile(ClientConnector cc, Map<String, String> attributes) throws Exception {
        log.debug("Starting FTP FilePull");
        String sourceNode = attributes.get("src_name");
        String sourceUser = attributes.get("src_user");
        String sourcePassword = attributes.get("src_password");
        String sourceFile = attributes.get("src_file");
        String messageID = attributes.get("messageID");
        String sourceMD5 = attributes.get("src_md5");
        String sourceFileType = attributes.get("src_file_type");
        Integer sourcePort = 21;
        String sourcePortString = attributes.get("src_port");
        if ((sourcePortString != null) && (sourcePortString.equals(""))) {
            try {
                sourcePort = Integer.parseInt(sourcePortString);
            } catch (Exception e) {
                sourcePort = 21;
                log.debug("Destination Port \"" + sourcePortString + "\" was not valid. Using Default (21)");
            }
        }
        log.info("Starting FTP pull of \"" + sourceFile + "\" from \"" + sourceNode);
        if ((sourceUser == null) || (sourceUser.equals(""))) {
            List userDBVal = axt.db.GeneralDAO.getNodeValue(sourceNode, "ftpUser");
            if (userDBVal.size() < 1) {
                sourceUser = DEFAULTUSER;
            } else {
                sourceUser = (String) userDBVal.get(0);
            }
        }
        if ((sourcePassword == null) || (sourcePassword.equals(""))) {
            List passwordDBVal = axt.db.GeneralDAO.getNodeValue(sourceNode, "ftpPassword");
            if (passwordDBVal.size() < 1) {
                sourcePassword = DEFAULTPASSWORD;
            } else {
                sourcePassword = (String) passwordDBVal.get(0);
            }
        }
        String stageFile = null;
        int stageFileID;
        try {
            stageFileID = axt.db.GeneralDAO.getStageFile(messageID);
            stageFile = STAGINGDIR + "/" + stageFileID;
        } catch (Exception e) {
            throw new Exception("Failed to assign a staging file \"" + stageFile + "\" - ERROR: " + e);
        }
        FileOutputStream fos;
        try {
            fos = new FileOutputStream(stageFile);
        } catch (FileNotFoundException fileNFException) {
            throw new Exception("Failed to assign the staging file \"" + stageFile + "\" - ERROR: " + fileNFException);
        }
        FTPClient ftp = new FTPClient();
        try {
            log.debug("Connecting");
            ftp.connect(sourceNode, sourcePort);
            log.debug("Checking Status");
            int reply = ftp.getReplyCode();
            if (!FTPReply.isPositiveCompletion(reply)) {
                ftp.disconnect();
                throw new Exception("Failed to connect to \"" + sourceNode + "\"  as user \"" + sourceUser + "\" - ERROR: " + ftp.getReplyString());
            }
            log.debug("Logging In");
            if (!ftp.login(sourceUser, sourcePassword)) {
                ftp.disconnect();
                throw new Exception("Failed to connect to \"" + sourceNode + "\"  as user \"" + sourceUser + "\" - ERROR: Login Failed");
            }
        } catch (SocketException socketException) {
            throw new Exception("Failed to connect to \"" + sourceNode + "\"  as user \"" + sourceUser + "\" - ERROR: " + socketException);
        } catch (IOException ioe) {
            throw new Exception("Failed to connect to \"" + sourceNode + "\"  as user \"" + sourceUser + "\" - ERROR: " + ioe);
        }
        log.debug("Performing Site Commands");
        Iterator siteIterator = GeneralDAO.getNodeValue(sourceNode, "ftpSite").iterator();
        while (siteIterator.hasNext()) {
            String siteCommand = null;
            try {
                siteCommand = (String) siteIterator.next();
                ftp.site(siteCommand);
            } catch (IOException e) {
                throw new Exception("FTP \"site\" command \"" + siteCommand + "\" failed - ERROR: " + e);
            }
        }
        if (sourceFileType != null) {
            if (sourceFileType.equals("A")) {
                log.debug("Set File Type to ASCII");
                ftp.setFileType(FTP.ASCII_FILE_TYPE);
            } else if (sourceFileType.equals("B")) {
                log.debug("Set File Type to BINARY");
                ftp.setFileType(FTP.BINARY_FILE_TYPE);
            } else if (sourceFileType.equals("E")) {
                log.debug("Set File Type to EBCDIC");
                ftp.setFileType(FTP.EBCDIC_FILE_TYPE);
            }
        }
        log.debug("Opening the File Stream");
        InputStream in = null;
        try {
            in = ftp.retrieveFileStream(sourceFile);
            if (in == null) {
                throw new Exception("Failed get the file \"" + sourceFile + "\" from \"" + sourceNode + "\"  - ERROR: " + ftp.getReplyString());
            }
        } catch (IOException ioe2) {
            ftp.disconnect();
            log.error("Failed get the file \"" + sourceFile + "\" from \"" + sourceNode + "\"  - ERROR: " + ioe2);
            throw new Exception("Failed to retrieve file from \"" + sourceNode + "\"  as user \"" + sourceUser + "\" - ERROR: " + ioe2);
        }
        log.debug("Starting the read");
        DESCrypt encrypter = null;
        try {
            encrypter = new DESCrypt();
        } catch (Exception cryptInitError) {
            log.error("Failed to initialize the encrypt process - ERROR: " + cryptInitError);
        }
        String receivedMD5 = null;
        try {
            Object[] returnValues = encrypter.encrypt(in, fos);
            receivedMD5 = (String) returnValues[0];
            GeneralDAO.setStageFileSize(stageFileID, (Long) returnValues[1]);
        } catch (Exception cryptError) {
            log.error("Encrypt Error: " + cryptError);
            throw new Exception("Encrypt Error: " + cryptError);
        }
        log.debug("Logging Out");
        try {
            ftp.logout();
            fos.close();
        } catch (Exception ioe3) {
            log.error("Failed close connection to \"" + sourceNode + "\"  - ERROR: " + ioe3);
        }
        log.debug("Setting the File Digest");
        GeneralDAO.setStageFileDigest(stageFileID, receivedMD5);
        if ((sourceMD5 != null) && (!sourceMD5.equals(""))) {
            log.debug("File DIGEST compare - Source: " + sourceMD5.toLowerCase() + " | Received: " + receivedMD5);
            if (!receivedMD5.equals(sourceMD5.toLowerCase())) {
                throw new Exception("MD5 validation on file failed.");
            }
        }
        return;
    }
```

## P020

**A**

```java
private static void createCompoundData(String dir, String type) {
        try {
            Set s = new HashSet();
            File nouns = new File(dir + "index." + type);
            FileInputStream fis = new FileInputStream(nouns);
            InputStreamReader reader = new InputStreamReader(fis);
            StringBuffer sb = new StringBuffer();
            int chr = reader.read();
            while (chr >= 0) {
                if (chr == '\n' || chr == '\r') {
                    String line = sb.toString();
                    if (line.length() > 0) {
                        String[] spaceSplit = PerlHelp.split(line);
                        for (int i = 0; i < spaceSplit.length; i++) {
                            if (spaceSplit[i].indexOf('_') >= 0) {
                                s.add(spaceSplit[i].replace('_', ' '));
                            }
                        }
                    }
                    sb.setLength(0);
                } else {
                    sb.append((char) chr);
                }
                chr = reader.read();
            }
            System.out.println(type + " size=" + s.size());
            File output = new File(dir + "compound." + type + "s.gz");
            FileOutputStream fos = new FileOutputStream(output);
            GZIPOutputStream gzos = new GZIPOutputStream(new BufferedOutputStream(fos));
            PrintWriter writer = new PrintWriter(gzos);
            writer.println("# This file was extracted from WordNet data, the following copyright notice");
            writer.println("# from WordNet is attached.");
            writer.println("#");
            writer.println("#  This software and database is being provided to you, the LICENSEE, by  ");
            writer.println("#  Princeton University under the following license.  By obtaining, using  ");
            writer.println("#  and/or copying this software and database, you agree that you have  ");
            writer.println("#  read, understood, and will comply with these terms and conditions.:  ");
            writer.println("#  ");
            writer.println("#  Permission to use, copy, modify and distribute this software and  ");
            writer.println("#  database and its documentation for any purpose and without fee or  ");
            writer.println("#  royalty is hereby granted, provided that you agree to comply with  ");
            writer.println("#  the following copyright notice and statements, including the disclaimer,  ");
            writer.println("#  and that the same appear on ALL copies of the software, database and  ");
            writer.println("#  documentation, including modifications that you make for internal  ");
            writer.println("#  use or for distribution.  ");
            writer.println("#  ");
            writer.println("#  WordNet 1.7 Copyright 2001 by Princeton University.  All rights reserved. ");
            writer.println("#  ");
            writer.println("#  THIS SOFTWARE AND DATABASE IS PROVIDED \"AS IS\" AND PRINCETON  ");
            writer.println("#  UNIVERSITY MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR  ");
            writer.println("#  IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, PRINCETON  ");
            writer.println("#  UNIVERSITY MAKES NO REPRESENTATIONS OR WARRANTIES OF MERCHANT-  ");
            writer.println("#  ABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE  ");
            writer.println("#  OF THE LICENSED SOFTWARE, DATABASE OR DOCUMENTATION WILL NOT  ");
            writer.println("#  INFRINGE ANY THIRD PARTY PATENTS, COPYRIGHTS, TRADEMARKS OR ");
            writer.println("#  OTHER RIGHTS. ");
            writer.println("#  ");
            writer.println("#  The name of Princeton University or Princeton may not be used in");
            writer.println("#  advertising or publicity pertaining to distribution of the software");
            writer.println("#  and/or database.  Title to copyright in this software, database and");
            writer.println("#  any associated documentation shall at all times remain with");
            writer.println("#  Princeton University and LICENSEE agrees to preserve same.  ");
            for (Iterator i = s.iterator(); i.hasNext(); ) {
                String mwe = (String) i.next();
                writer.println(mwe);
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
```

**B**

```java
@SuppressWarnings("unchecked")
    private List<String> getWordList() {
        IConfiguration config = Configurator.getDefaultConfigurator().getConfig(CONFIG_ID);
        List<String> wList = (List<String>) config.getObject("word_list");
        if (wList == null) {
            wList = new ArrayList<String>();
            InputStream resrc = null;
            try {
                resrc = new URL(list_url).openStream();
            } catch (Exception e) {
                e.printStackTrace();
            }
            if (resrc != null) {
                BufferedReader br = new BufferedReader(new InputStreamReader(resrc));
                String line;
                try {
                    while ((line = br.readLine()) != null) {
                        line = line.trim();
                        if (line.length() != 0) {
                            wList.add(line);
                        }
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                } finally {
                    if (br != null) {
                        try {
                            br.close();
                        } catch (IOException e) {
                        }
                    }
                }
            }
        }
        return wList;
    }
```

## P021

**A**

```java
public int addPermissionsForUserAndAgenda(Integer userId, Integer agendaId, String permissions) throws TechnicalException {
        if (permissions == null) {
            throw new TechnicalException(new Exception(new Exception("Column 'permissions' cannot be null")));
        }
        Session session = null;
        Transaction transaction = null;
        try {
            session = HibernateUtil.getCurrentSession();
            transaction = session.beginTransaction();
            String query = "INSERT INTO j_user_agenda (userId, agendaId, permissions) VALUES(" + userId + "," + agendaId + ",\"" + permissions + "\")";
            Statement statement = session.connection().createStatement();
            int rowsUpdated = statement.executeUpdate(query);
            transaction.commit();
            return rowsUpdated;
        } catch (HibernateException ex) {
            if (transaction != null) transaction.rollback();
            throw new TechnicalException(ex);
        } catch (SQLException e) {
            if (transaction != null) transaction.rollback();
            throw new TechnicalException(e);
        }
    }
```

**B**

```java
public Program updateProgramPath(int id, String sourcePath) throws AdaptationException {
        Program program = null;
        Connection connection = null;
        Statement statement = null;
        ResultSet resultSet = null;
        try {
            String query = "UPDATE Programs SET " + "sourcePath = '" + sourcePath + "' " + "WHERE id = " + id;
            connection = DriverManager.getConnection(CONN_STR);
            statement = connection.createStatement();
            statement.executeUpdate(query);
            query = "SELECT * from Programs WHERE id = " + id;
            resultSet = statement.executeQuery(query);
            if (!resultSet.next()) {
                connection.rollback();
                String msg = "Attempt to update program failed.";
                log.error(msg);
                throw new AdaptationException(msg);
            }
            program = getProgram(resultSet);
            connection.commit();
        } catch (SQLException ex) {
            try {
                connection.rollback();
            } catch (Exception e) {
            }
            String msg = "SQLException in updateProgramPath";
            log.error(msg, ex);
            throw new AdaptationException(msg, ex);
        } finally {
            try {
                resultSet.close();
            } catch (Exception ex) {
            }
            try {
                statement.close();
            } catch (Exception ex) {
            }
            try {
                connection.close();
            } catch (Exception ex) {
            }
        }
        return program;
    }
```

## P022

**A**

```java
private void copyJdbcDriverToWL(final WLPropertyPage page) {
        final File url = new File(page.getDomainDirectory());
        final File lib = new File(url, "lib");
        final File mysqlLibrary = new File(lib, NexOpenUIActivator.getDefault().getMySQLDriver());
        if (!mysqlLibrary.exists()) {
            InputStream driver = null;
            FileOutputStream fos = null;
            try {
                driver = AppServerPropertyPage.toInputStream(new Path("jdbc/" + NexOpenUIActivator.getDefault().getMySQLDriver()));
                fos = new FileOutputStream(mysqlLibrary);
                IOUtils.copy(driver, fos);
            } catch (final IOException e) {
                Logger.log(Logger.ERROR, "Could not copy the MySQL Driver jar file to Bea WL", e);
                final Status status = new Status(Status.ERROR, NexOpenUIActivator.PLUGIN_ID, Status.ERROR, "Could not copy the MySQL Driver jar file to Bea WL", e);
                ErrorDialog.openError(page.getShell(), "Bea WebLogic MSQL support", "Could not copy the MySQL Driver jar file to Bea WL", status);
            } finally {
                try {
                    if (driver != null) {
                        driver.close();
                        driver = null;
                    }
                    if (fos != null) {
                        fos.flush();
                        fos.close();
                        fos = null;
                    }
                } catch (IOException e) {
                }
            }
        }
    }
```

**B**

```java
private WikiSiteContentInfo createInfoIndexSite(Long domainId) {
        final UserInfo user = getSecurityService().getCurrentUser();
        final Locale locale = new Locale(user.getLocale());
        final String country = locale.getLanguage();
        InputStream inStream = Thread.currentThread().getContextClassLoader().getResourceAsStream("wiki_index_" + country + ".xhtml");
        if (inStream == null) {
            inStream = Thread.currentThread().getContextClassLoader().getResourceAsStream("wiki_index.xhtml");
        }
        if (inStream == null) {
            inStream = new ByteArrayInputStream(DEFAULT_WIKI_INDEX_SITE_TEXT.getBytes());
        }
        if (inStream != null) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            try {
                IOUtils.copyLarge(inStream, out);
                return createIndexVersion(domainId, out.toString(), user);
            } catch (IOException exception) {
                LOGGER.error("Error creating info page.", exception);
            } finally {
                try {
                    inStream.close();
                    out.close();
                } catch (IOException exception) {
                    LOGGER.error("Error reading wiki_index.xhtml", exception);
                }
            }
        }
        return null;
    }
```

## P023

**A**

```java
private static String doGetForSessionKey(String authCode) throws Exception {
        String sessionKey = "";
        HttpClient hc = new DefaultHttpClient();
        HttpGet hg = new HttpGet(Common.TEST_SESSION_HOST + Common.TEST_SESSION_PARAM + authCode);
        HttpResponse hr = hc.execute(hg);
        BufferedReader br = new BufferedReader(new InputStreamReader(hr.getEntity().getContent()));
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = br.readLine()) != null) {
            sb.append(line);
        }
        String result = sb.toString();
        Log.i("sessionKeyMessages", result);
        Map<String, String> map = Util.handleURLParameters(result);
        sessionKey = map.get(Common.TOP_SESSION);
        String topParameters = map.get(Common.TOP_PARAMETERS);
        String decTopParameters = Util.decodeBase64(topParameters);
        Log.i("base64", decTopParameters);
        map = Util.handleURLParameters(decTopParameters);
        Log.i("nick", map.get(Common.VISITOR_NICK));
        CachePool.put(Common.VISITOR_NICK, map.get(Common.VISITOR_NICK));
        return sessionKey;
    }
```

**B**

```java
public static String upLoadImg(File pic, String uid) throws Throwable {
        System.out.println("开始上传=======================================================");
        HttpPost post = getHttpPost(getUploadUrl(uid), uid);
        FileBody file = new FileBody(pic, "image/jpg");
        MultipartEntity reqEntity = new MultipartEntity();
        reqEntity.addPart("pic1", file);
        post.setEntity(reqEntity);
        HttpResponse response = client.execute(post);
        int status = response.getStatusLine().getStatusCode();
        post.abort();
        if (status == HttpStatus.SC_MOVED_TEMPORARILY || status == HttpStatus.SC_MOVED_PERMANENTLY) {
            String newuri = response.getHeaders("location")[0].getValue();
            System.out.println(newuri);
            return newuri.substring(newuri.indexOf("pid=") + 4, newuri.indexOf("&token="));
        }
        return null;
    }
```

## P024

**A**

```java
public static void fixEol(File fin) throws IOException {
        File fout = File.createTempFile(fin.getName(), ".fixEol", fin.getParentFile());
        FileChannel in = new FileInputStream(fin).getChannel();
        if (0 != in.size()) {
            FileChannel out = new FileOutputStream(fout).getChannel();
            byte[] eol = AStringUtilities.systemNewLine.getBytes();
            ByteBuffer bufOut = ByteBuffer.allocateDirect(1024 * eol.length);
            boolean previousIsCr = false;
            ByteBuffer buf = ByteBuffer.allocateDirect(1024);
            while (in.read(buf) > 0) {
                buf.limit(buf.position());
                buf.position(0);
                while (buf.remaining() > 0) {
                    byte b = buf.get();
                    if (b == '\r') {
                        previousIsCr = true;
                        bufOut.put(eol);
                    } else {
                        if (b == '\n') {
                            if (!previousIsCr) bufOut.put(eol);
                        } else bufOut.put(b);
                        previousIsCr = false;
                    }
                }
                bufOut.limit(bufOut.position());
                bufOut.position(0);
                out.write(bufOut);
                bufOut.clear();
                buf.clear();
            }
            out.close();
        }
        in.close();
        fin.delete();
        fout.renameTo(fin);
    }
```

**B**

```java
public static void copyFile(File src, File dest, boolean notifyUserOnError) {
        if (src.exists()) {
            try {
                BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(dest));
                BufferedInputStream in = new BufferedInputStream(new FileInputStream(src));
                byte[] read = new byte[128];
                int len = 128;
                while ((len = in.read(read)) > 0) out.write(read, 0, len);
                out.flush();
                out.close();
                in.close();
            } catch (IOException e) {
                String message = "Error while copying " + src.getAbsolutePath() + " to " + dest.getAbsolutePath() + " : " + e.getMessage();
                if (notifyUserOnError) {
                    Log.getInstance(SystemUtils.class).warnWithUserNotification(message);
                } else {
                    Log.getInstance(SystemUtils.class).warn(message);
                }
            }
        } else {
            String message = "Unable to copy file: source does not exists: " + src.getAbsolutePath();
            if (notifyUserOnError) {
                Log.getInstance(SystemUtils.class).warnWithUserNotification(message);
            } else {
                Log.getInstance(SystemUtils.class).warn(message);
            }
        }
    }
```

## P025

**A**

```java
protected void checkWeavingJar() throws IOException {
        OutputStream out = null;
        try {
            final File weaving = new File(getWeavingPath());
            if (!weaving.exists()) {
                new File(getWeavingFolder()).mkdir();
                weaving.createNewFile();
                final Path src = new Path("weaving/openfrwk-weaving.jar");
                final InputStream in = FileLocator.openStream(getBundle(), src, false);
                out = new FileOutputStream(getWeavingPath(), true);
                IOUtils.copy(in, out);
                Logger.log(Logger.INFO, "Put weaving jar at location " + weaving);
            } else {
                Logger.getLog().info("File openfrwk-weaving.jar already exists at " + weaving);
            }
        } catch (final SecurityException e) {
            Logger.log(Logger.ERROR, "[SECURITY EXCEPTION] Not enough privilegies to create " + "folder and copy NexOpen weaving jar at location " + getWeavingFolder());
            Logger.logException(e);
        } finally {
            if (out != null) {
                out.flush();
                out.close();
            }
        }
    }
```

**B**

```java
protected void find(final String pckgname, final boolean recursive) {
            URL url;
            String name = pckgname;
            name = name.replace('.', '/');
            url = ResourceLocatorTool.getClassPathResource(ExampleRunner.class, name);
            File directory;
            try {
                directory = new File(URLDecoder.decode(url.getFile(), "UTF-8"));
            } catch (final UnsupportedEncodingException e) {
                throw new RuntimeException(e);
            }
            if (directory.exists()) {
                logger.info("Searching for examples in \"" + directory.getPath() + "\".");
                addAllFilesInDirectory(directory, pckgname, recursive);
            } else {
                try {
                    logger.info("Searching for Demo classes in \"" + url + "\".");
                    final URLConnection urlConnection = url.openConnection();
                    if (urlConnection instanceof JarURLConnection) {
                        final JarURLConnection conn = (JarURLConnection) urlConnection;
                        final JarFile jfile = conn.getJarFile();
                        final Enumeration<JarEntry> e = jfile.entries();
                        while (e.hasMoreElements()) {
                            final ZipEntry entry = e.nextElement();
                            final Class<?> result = load(entry.getName());
                            if (result != null) {
                                addClassForPackage(result);
                            }
                        }
                    }
                } catch (final IOException e) {
                    logger.logp(Level.SEVERE, this.getClass().toString(), "find(pckgname, recursive, classes)", "Exception", e);
                } catch (final Exception e) {
                    logger.logp(Level.SEVERE, this.getClass().toString(), "find(pckgname, recursive, classes)", "Exception", e);
                }
            }
        }
```

## P026

**A**

```java
public String doAdd(ActionMapping mapping, ActionForm form, HttpServletRequest request, HttpServletResponse response) throws Exception {
        if (logger.isDebugEnabled()) {
            logger.debug("doAdd(ActionMapping, ActionForm, HttpServletRequest, HttpServletResponse) - start");
        }
        t_information_EditMap editMap = new t_information_EditMap();
        try {
            t_information_Form vo = null;
            vo = (t_information_Form) form;
            vo.setCompany(vo.getCounty());
            if ("����".equals(vo.getInfo_type())) {
                vo.setInfo_level(null);
                vo.setAlert_level(null);
            }
            String str_postFIX = "";
            int i_p = 0;
            editMap.add(vo);
            try {
                logger.info("���͹�˾�鱨��");
                String[] mobiles = request.getParameterValues("mobiles");
                vo.setMobiles(mobiles);
                SMSService.inforAlert(vo);
            } catch (Exception e) {
                logger.error("doAdd(ActionMapping, ActionForm, HttpServletRequest, HttpServletResponse)", e);
            }
            String filename = vo.getFile().getFileName();
            if (null != filename && !"".equals(filename)) {
                FormFile file = vo.getFile();
                String realpath = getServlet().getServletContext().getRealPath("/");
                realpath = realpath.replaceAll("\\\\", "/");
                String inforId = vo.getId();
                String rootFilePath = getServlet().getServletContext().getRealPath(request.getContextPath());
                rootFilePath = (new StringBuilder(String.valueOf(rootFilePath))).append(UploadFileOne.strPath).toString();
                String strAppend = (new StringBuilder(String.valueOf(UUIDGenerator.nextHex()))).append(UploadFileOne.getFileType(file)).toString();
                if (file.getFileSize() != 0) {
                    file.getInputStream();
                    String name = file.getFileName();
                    i_p = file.getFileName().lastIndexOf(".");
                    str_postFIX = file.getFileName().substring(i_p, file.getFileName().length());
                    String fullPath = realpath + "attach/" + strAppend + str_postFIX;
                    t_attach attach = new t_attach();
                    attach.setAttach_fullname(fullPath);
                    attach.setAttach_name(name);
                    attach.setInfor_id(Integer.parseInt(inforId));
                    attach.setInsert_day(new Date());
                    attach.setUpdate_day(new Date());
                    t_attach_EditMap attachEdit = new t_attach_EditMap();
                    attachEdit.add(attach);
                    File sysfile = new File(fullPath);
                    if (!sysfile.exists()) {
                        sysfile.createNewFile();
                    }
                    java.io.OutputStream out = new FileOutputStream(sysfile);
                    org.apache.commons.io.IOUtils.copy(file.getInputStream(), out);
                    out.close();
                }
            }
        } catch (HibernateException e) {
            logger.error("doAdd(ActionMapping, ActionForm, HttpServletRequest, HttpServletResponse)", e);
            ActionErrors errors = new ActionErrors();
            errors.add("org.apache.struts.action.GLOBAL_ERROR", new ActionError("error.database.save", e.toString()));
            saveErrors(request, errors);
            e.printStackTrace();
            request.setAttribute("t_information_Form", form);
            if (logger.isDebugEnabled()) {
                logger.debug("doAdd(ActionMapping, ActionForm, HttpServletRequest, HttpServletResponse) - end");
            }
            return "addpage";
        }
        if (logger.isDebugEnabled()) {
            logger.debug("doAdd(ActionMapping, ActionForm, HttpServletRequest, HttpServletResponse) - end");
        }
        return "aftersave";
    }
```

**B**

```java
public ActionForward uploadFile(ActionMapping mapping, ActionForm actForm, HttpServletRequest request, HttpServletResponse in_response) {
        ActionMessages errors = new ActionMessages();
        ActionMessages messages = new ActionMessages();
        String returnPage = "submitPocketSampleInformationPage";
        UploadForm form = (UploadForm) actForm;
        Integer shippingId = null;
        try {
            eHTPXXLSParser parser = new eHTPXXLSParser();
            String proposalCode;
            String proposalNumber;
            String proposalName;
            String uploadedFileName;
            String realXLSPath;
            if (request != null) {
                proposalCode = (String) request.getSession().getAttribute(Constants.PROPOSAL_CODE);
                proposalNumber = String.valueOf(request.getSession().getAttribute(Constants.PROPOSAL_NUMBER));
                proposalName = proposalCode + proposalNumber.toString();
                uploadedFileName = form.getRequestFile().getFileName();
                String fileName = proposalName + "_" + uploadedFileName;
                realXLSPath = request.getRealPath("\\tmp\\") + "\\" + fileName;
                FormFile f = form.getRequestFile();
                InputStream in = f.getInputStream();
                File outputFile = new File(realXLSPath);
                if (outputFile.exists()) outputFile.delete();
                FileOutputStream out = new FileOutputStream(outputFile);
                while (in.available() != 0) {
                    out.write(in.read());
                    out.flush();
                }
                out.flush();
                out.close();
            } else {
                proposalCode = "ehtpx";
                proposalNumber = "1";
                proposalName = proposalCode + proposalNumber.toString();
                uploadedFileName = "ispyb-template41.xls";
                realXLSPath = "D:\\" + uploadedFileName;
            }
            FileInputStream inFile = new FileInputStream(realXLSPath);
            parser.retrieveShippingId(realXLSPath);
            shippingId = parser.getShippingId();
            String requestShippingId = form.getShippingId();
            if (requestShippingId != null && !requestShippingId.equals("")) {
                shippingId = new Integer(requestShippingId);
            }
            ClientLogger.getInstance().debug("uploadFile for shippingId " + shippingId);
            if (shippingId != null) {
                Log.debug(" ---[uploadFile] Upload for Existing Shipment (DewarTRacking): Deleting Samples from Shipment :");
                double nbSamplesContainers = DBAccess_EJB.DeleteAllSamplesAndContainersForShipping(shippingId);
                if (nbSamplesContainers > 0) parser.getValidationWarnings().add(new XlsUploadException("Shipment contained Samples and/or Containers", "Previous Samples and/or Containers have been deleted and replaced by new ones.")); else parser.getValidationWarnings().add(new XlsUploadException("Shipment contained no Samples and no Containers", "Samples and Containers have been added."));
            }
            Hashtable<String, Hashtable<String, Integer>> listProteinAcronym_SampleName = new Hashtable<String, Hashtable<String, Integer>>();
            ProposalFacadeLocal proposal = ProposalFacadeUtil.getLocalHome().create();
            ProteinFacadeLocal protein = ProteinFacadeUtil.getLocalHome().create();
            CrystalFacadeLocal crystal = CrystalFacadeUtil.getLocalHome().create();
            ProposalLightValue targetProposal = (ProposalLightValue) (((ArrayList) proposal.findByCodeAndNumber(proposalCode, new Integer(proposalNumber))).get(0));
            ArrayList listProteins = (ArrayList) protein.findByProposalId(targetProposal.getProposalId());
            for (int p = 0; p < listProteins.size(); p++) {
                ProteinValue prot = (ProteinValue) listProteins.get(p);
                Hashtable<String, Integer> listSampleName = new Hashtable<String, Integer>();
                CrystalLightValue listCrystals[] = prot.getCrystals();
                for (int c = 0; c < listCrystals.length; c++) {
                    CrystalLightValue _xtal = (CrystalLightValue) listCrystals[c];
                    CrystalValue xtal = crystal.findByPrimaryKey(_xtal.getPrimaryKey());
                    BlsampleLightValue listSamples[] = xtal.getBlsamples();
                    for (int s = 0; s < listSamples.length; s++) {
                        BlsampleLightValue sample = listSamples[s];
                        listSampleName.put(sample.getName(), sample.getBlSampleId());
                    }
                }
                listProteinAcronym_SampleName.put(prot.getAcronym(), listSampleName);
            }
            parser.validate(inFile, listProteinAcronym_SampleName, targetProposal.getProposalId());
            List listErrors = parser.getValidationErrors();
            List listWarnings = parser.getValidationWarnings();
            if (listErrors.size() == 0) {
                parser.open(realXLSPath);
                if (parser.getCrystals().size() == 0) {
                    parser.getValidationErrors().add(new XlsUploadException("No crystals have been found", "Empty shipment"));
                }
            }
            Iterator errIt = listErrors.iterator();
            while (errIt.hasNext()) {
                XlsUploadException xlsEx = (XlsUploadException) errIt.next();
                errors.add(ActionMessages.GLOBAL_MESSAGE, new ActionMessage("message.free", xlsEx.getMessage() + " ---> " + xlsEx.getSuggestedFix()));
            }
            try {
                saveErrors(request, errors);
            } catch (Exception e) {
            }
            Iterator warnIt = listWarnings.iterator();
            while (warnIt.hasNext()) {
                XlsUploadException xlsEx = (XlsUploadException) warnIt.next();
                messages.add(ActionMessages.GLOBAL_MESSAGE, new ActionMessage("message.free", xlsEx.getMessage() + " ---> " + xlsEx.getSuggestedFix()));
            }
            try {
                saveMessages(request, messages);
            } catch (Exception e) {
            }
            if (listErrors.size() > 0) {
                resetCounts(shippingId);
                return mapping.findForward("submitPocketSampleInformationPage");
            }
            if (listWarnings.size() > 0) returnPage = "submitPocketSampleInformationPage";
            String crystalDetailsXML;
            XtalDetails xtalDetailsWebService = new XtalDetails();
            CrystalDetailsBuilder cDE = new CrystalDetailsBuilder();
            CrystalDetailsElement cd = cDE.createCrystalDetailsElement(proposalName, parser.getCrystals());
            cDE.validateJAXBObject(cd);
            crystalDetailsXML = cDE.marshallJaxBObjToString(cd);
            xtalDetailsWebService.submitCrystalDetails(crystalDetailsXML);
            String diffractionPlan;
            DiffractionPlan diffractionPlanWebService = new DiffractionPlan();
            DiffractionPlanBuilder dPB = new DiffractionPlanBuilder();
            Iterator it = parser.getDiffractionPlans().iterator();
            while (it.hasNext()) {
                DiffractionPlanElement dpe = (DiffractionPlanElement) it.next();
                dpe.setProjectUUID(proposalName);
                diffractionPlan = dPB.marshallJaxBObjToString(dpe);
                diffractionPlanWebService.submitDiffractionPlan(diffractionPlan);
            }
            String crystalShipping;
            Shipping shippingWebService = new Shipping();
            CrystalShippingBuilder cSB = new CrystalShippingBuilder();
            Person person = cSB.createPerson("XLS Upload", null, "ISPyB", null, null, "ISPyB", null, "ispyb@esrf.fr", "0000", "0000", null, null);
            Laboratory laboratory = cSB.createLaboratory("Generic Laboratory", "ISPyB Lab", "Sandwich", "Somewhere", "UK", "ISPyB", "ispyb.esrf.fr", person);
            DeliveryAgent deliveryAgent = parser.getDeliveryAgent();
            CrystalShipping cs = cSB.createCrystalShipping(proposalName, laboratory, deliveryAgent, parser.getDewars());
            String shippingName;
            shippingName = uploadedFileName.substring(0, ((uploadedFileName.toLowerCase().lastIndexOf(".xls")) > 0) ? uploadedFileName.toLowerCase().lastIndexOf(".xls") : 0);
            if (shippingName.equalsIgnoreCase("")) shippingName = uploadedFileName.substring(0, ((uploadedFileName.toLowerCase().lastIndexOf(".xlt")) > 0) ? uploadedFileName.toLowerCase().lastIndexOf(".xlt") : 0);
            cs.setName(shippingName);
            crystalShipping = cSB.marshallJaxBObjToString(cs);
            shippingWebService.submitCrystalShipping(crystalShipping, (ArrayList) parser.getDiffractionPlans(), shippingId);
            ServerLogger.Log4Stat("XLS_UPLOAD", proposalName, uploadedFileName);
        } catch (XlsUploadException e) {
            resetCounts(shippingId);
            errors.add(ActionMessages.GLOBAL_MESSAGE, new ActionMessage("errors.detail", e.getMessage()));
            ClientLogger.getInstance().error(e.toString());
            saveErrors(request, errors);
            return mapping.findForward("error");
        } catch (Exception e) {
            resetCounts(shippingId);
            errors.add(ActionMessages.GLOBAL_MESSAGE, new ActionMessage("errors.detail", e.toString()));
            ClientLogger.getInstance().error(e.toString());
            e.printStackTrace();
            saveErrors(request, errors);
            return mapping.findForward("error");
        }
        setCounts(shippingId);
        return mapping.findForward(returnPage);
    }
```

## P027

**A**

```java
public void getZipFiles(String filename) {
        try {
            String destinationname = "c:\\mods\\peu\\";
            byte[] buf = new byte[1024];
            ZipInputStream zipinputstream = null;
            ZipEntry zipentry;
            zipinputstream = new ZipInputStream(new FileInputStream(filename));
            zipentry = zipinputstream.getNextEntry();
            while (zipentry != null) {
                String entryName = zipentry.getName();
                System.out.println("entryname " + entryName);
                int n;
                FileOutputStream fileoutputstream;
                File newFile = new File(entryName);
                String directory = newFile.getParent();
                if (directory == null) {
                    if (newFile.isDirectory()) break;
                }
                fileoutputstream = new FileOutputStream(destinationname + entryName);
                while ((n = zipinputstream.read(buf, 0, 1024)) > -1) fileoutputstream.write(buf, 0, n);
                fileoutputstream.close();
                zipinputstream.closeEntry();
                zipentry = zipinputstream.getNextEntry();
            }
            zipinputstream.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
```

**B**

```java
public static void unzipModel(String filename, String tempdir) throws Exception {
        try {
            BufferedOutputStream dest = null;
            FileInputStream fis = new FileInputStream(filename);
            int BUFFER = 2048;
            ZipInputStream zis = new ZipInputStream(new BufferedInputStream(fis));
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                int count;
                byte data[] = new byte[BUFFER];
                FileOutputStream fos = new FileOutputStream(tempdir + entry.getName());
                dest = new BufferedOutputStream(fos, BUFFER);
                while ((count = zis.read(data, 0, BUFFER)) != -1) dest.write(data, 0, count);
                dest.flush();
                dest.close();
            }
            zis.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new Exception("Can not expand model in \"" + tempdir + "\" because:\n" + e.getMessage());
        }
    }
```

## P028

**A**

```java
private void documentFileChooserActionPerformed(java.awt.event.ActionEvent evt) {
        if (evt.getActionCommand().equals(JFileChooser.APPROVE_SELECTION)) {
            File selectedFile = documentFileChooser.getSelectedFile();
            File collectionCopyFile;
            String newDocumentName = selectedFile.getName();
            Document newDocument = new Document(newDocumentName);
            if (activeCollection.containsDocument(newDocument)) {
                int matchingFilenameDistinguisher = 1;
                StringBuilder distinguisherReplacer = new StringBuilder();
                newDocumentName = newDocumentName.concat("(" + matchingFilenameDistinguisher + ")");
                newDocument.setDocumentName(newDocumentName);
                while (activeCollection.containsDocument(newDocument)) {
                    matchingFilenameDistinguisher++;
                    newDocumentName = distinguisherReplacer.replace(newDocumentName.length() - 2, newDocumentName.length() - 1, new Integer(matchingFilenameDistinguisher).toString()).toString();
                    newDocument.setDocumentName(newDocumentName);
                }
            }
            Scanner tokenizer = null;
            FileChannel fileSource = null;
            FileChannel collectionDestination = null;
            HashMap<String, Integer> termHashMap = new HashMap<String, Integer>();
            Index collectionIndex = activeCollection.getIndex();
            int documentTermMaxFrequency = 0;
            int currentTermFrequency;
            try {
                tokenizer = new Scanner(new BufferedReader(new FileReader(selectedFile)));
                tokenizer.useDelimiter(Pattern.compile("\\p{Space}|\\p{Punct}|\\p{Cntrl}"));
                String nextToken;
                while (tokenizer.hasNext()) {
                    nextToken = tokenizer.next().toLowerCase();
                    if (!nextToken.isEmpty()) if (termHashMap.containsKey(nextToken)) termHashMap.put(nextToken, termHashMap.get(nextToken) + 1); else termHashMap.put(nextToken, 1);
                }
                Term newTerm;
                for (String term : termHashMap.keySet()) {
                    newTerm = new Term(term);
                    if (!collectionIndex.termExists(newTerm)) collectionIndex.addTerm(newTerm);
                    currentTermFrequency = termHashMap.get(term);
                    if (currentTermFrequency > documentTermMaxFrequency) documentTermMaxFrequency = currentTermFrequency;
                    collectionIndex.addOccurence(newTerm, newDocument, currentTermFrequency);
                }
                activeCollection.addDocument(newDocument);
                String userHome = System.getProperty("user.home");
                String fileSeparator = System.getProperty("file.separator");
                collectionCopyFile = new File(userHome + fileSeparator + "Infrared" + fileSeparator + activeCollection.getDocumentCollectionName() + fileSeparator + newDocumentName);
                collectionCopyFile.createNewFile();
                fileSource = new FileInputStream(selectedFile).getChannel();
                collectionDestination = new FileOutputStream(collectionCopyFile).getChannel();
                collectionDestination.transferFrom(fileSource, 0, fileSource.size());
            } catch (FileNotFoundException e) {
                System.err.println(e.getMessage() + " This error should never occur! The file was just selected!");
                return;
            } catch (IOException e) {
                JOptionPane.showMessageDialog(this, "An I/O error occured during file transfer!", "File transfer I/O error", JOptionPane.WARNING_MESSAGE);
                return;
            } finally {
                try {
                    if (tokenizer != null) tokenizer.close();
                    if (fileSource != null) fileSource.close();
                    if (collectionDestination != null) collectionDestination.close();
                } catch (IOException e) {
                    System.err.println(e.getMessage());
                }
            }
            processWindowEvent(new WindowEvent(this, WindowEvent.WINDOW_CLOSING));
        } else if (evt.getActionCommand().equalsIgnoreCase(JFileChooser.CANCEL_SELECTION)) processWindowEvent(new WindowEvent(this, WindowEvent.WINDOW_CLOSING));
    }
```

**B**

```java
protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        session = request.getSession(true);
        response.setContentType("text/html");
        response.setCharacterEncoding("UTF-8");
        PrintWriter out = response.getWriter();
        try {
            String searchTerm = new String();
            if (request.getParameter("searchdb") != null) {
                searchTerm = request.getParameter("searchdb");
                out.write("<ul>");
                PreparedStatement sqlGetLikeBaseString = conn.prepareStatement("SELECT * FROM ENTRIES WHERE XTM_SESSION_ID = ? AND XTM_TEXT LIKE ?");
                sqlGetLikeBaseString.setString(1, session.getId());
                sqlGetLikeBaseString.setString(2, new String("%" + searchTerm + "%"));
                ResultSet res = sqlGetLikeBaseString.executeQuery();
                while (res.next()) {
                    out.write("<li>");
                    out.write(res.getString("XTM_TEXT"));
                    out.write("</li>");
                }
                out.write("</ul>");
                res.close();
            }
            if (request.getParameter("searchwiki") != null) {
                searchTerm = request.getParameter("searchwiki");
                out.write("<ul>");
                try {
                    searchTerm = URLEncoder.encode(searchTerm, "UTF-8");
                    URL url = new URL("http://www.wikipedia.de/suggest.php?lang=de&search=" + searchTerm);
                    URLConnection con = url.openConnection();
                    BufferedReader rd = new BufferedReader(new InputStreamReader(con.getInputStream(), "UTF-8"));
                    String line;
                    while ((line = rd.readLine()) != null) {
                        out.write("<li>");
                        String[] split = line.split("\t");
                        out.write(split[0]);
                        out.write("</li>");
                    }
                    rd.close();
                } catch (Exception e) {
                    e.printStackTrace();
                }
                out.write("</ul>");
            } else {
                return;
            }
        } catch (SQLException e) {
            out.println("Caught SQLException:" + e.getMessage());
        }
        ;
    }
```

## P029

**A**

```java
@Test
    public void testCopy_readerToOutputStream_nullOut() throws Exception {
        InputStream in = new ByteArrayInputStream(inData);
        in = new YellOnCloseInputStreamTest(in);
        Reader reader = new InputStreamReader(in, "US-ASCII");
        try {
            IOUtils.copy(reader, (OutputStream) null);
            fail();
        } catch (NullPointerException ex) {
        }
    }
```

**B**

```java
public static void main(String[] args) throws Exception {
        String uri = args[0];
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(URI.create(uri), conf);
        InputStream in = null;
        try {
            in = fs.open(new Path(uri));
            IOUtils.copyBytes(in, System.out, 4096, false);
        } finally {
            IOUtils.closeStream(in);
        }
    }
```

## P030

**A**

```java
@Override
    protected Integer doInBackground() throws Exception {
        int numOfRows = 0;
        combinationMap = new HashMap<AnsweredQuestion, Integer>();
        combinationMapReverse = new HashMap<Integer, AnsweredQuestion>();
        LinkedHashSet<AnsweredQuestion> answeredQuestionSet = new LinkedHashSet<AnsweredQuestion>();
        LinkedHashSet<Integer> studentSet = new LinkedHashSet<Integer>();
        final String delimiter = ";";
        final String typeToProcess = "F";
        String line;
        String[] chunks = new String[9];
        try {
            BufferedReader in = new BufferedReader(new InputStreamReader(url.openStream(), "ISO-8859-2"));
            in.readLine();
            while ((line = in.readLine()) != null) {
                chunks = line.split(delimiter);
                numOfRows++;
                if (chunks[2].equals(typeToProcess)) {
                    answeredQuestionSet.add(new AnsweredQuestion(chunks[4], chunks[5]));
                    studentSet.add(new Integer(chunks[0]));
                }
            }
            in.close();
            int i = 0;
            Integer I;
            for (AnsweredQuestion pair : answeredQuestionSet) {
                I = new Integer(i++);
                combinationMap.put(pair, I);
                combinationMapReverse.put(I, pair);
            }
            matrix = new SparseObjectMatrix2D(answeredQuestionSet.size(), studentSet.size());
            int lastStudentNumber = -1;
            AnsweredQuestion pair;
            in = new BufferedReader(new InputStreamReader(url.openStream(), "ISO-8859-2"));
            in.readLine();
            while ((line = in.readLine()) != null) {
                chunks = line.split(delimiter);
                pair = null;
                if (chunks[2].equals(typeToProcess)) {
                    if (Integer.parseInt(chunks[0]) != lastStudentNumber) {
                        lastStudentNumber++;
                    }
                    pair = new AnsweredQuestion(chunks[4], chunks[5]);
                    if (combinationMap.containsKey(pair)) {
                        matrix.setQuick(combinationMap.get(pair), lastStudentNumber, Boolean.TRUE);
                    }
                }
            }
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        supportVector = new int[combinationMap.size()];
        ObjectMatrix1D row = null;
        for (int i = 0; i < combinationMap.size(); i++) {
            row = matrix.viewRow(i);
            int sum = 0;
            for (int k = 0; k < row.size(); k++) {
                if (row.getQuick(k) != null && row.getQuick(k).equals(Boolean.TRUE)) {
                    sum++;
                }
            }
            supportVector[i] = sum;
        }
        applet.combinationMap = this.combinationMap;
        applet.combinationMapReverse = this.combinationMapReverse;
        applet.matrix = this.matrix;
        applet.supportVector = supportVector;
        System.out.println("data loaded.");
        return null;
    }
```

**B**

```java
public LinkedList<NameValuePair> getQuestion() {
        InputStream is = null;
        String result = "";
        LinkedList<NameValuePair> question = new LinkedList<NameValuePair>();
        try {
            HttpClient httpclient = new DefaultHttpClient();
            HttpPost httppost = new HttpPost(domain);
            httppost.setEntity(new UrlEncodedFormEntity(library));
            HttpResponse response = httpclient.execute(httppost);
            HttpEntity entity = response.getEntity();
            is = entity.getContent();
        } catch (Exception e) {
            Log.e("log_tag", "Error in http connection " + e.toString());
        }
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(is, "iso-8859-1"), 8);
            StringBuilder sb = new StringBuilder();
            String line = null;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
            is.close();
            result = sb.toString();
            if (result.equals("null,")) {
                return null;
            }
        } catch (Exception e) {
            Log.e("log_tag", "Error converting result " + e.toString());
        }
        try {
            JSONObject json = new JSONObject(result);
            JSONArray data = json.getJSONArray("data");
            JSONObject quest = data.getJSONObject(0);
            question.add(new BasicNameValuePair("q", quest.getString("q")));
            question.add(new BasicNameValuePair("a", quest.getString("a")));
            question.add(new BasicNameValuePair("b", quest.getString("b")));
            question.add(new BasicNameValuePair("c", quest.getString("c")));
            question.add(new BasicNameValuePair("d", quest.getString("d")));
            question.add(new BasicNameValuePair("correct", quest.getString("correct")));
            return question;
        } catch (JSONException e) {
            Log.e("log_tag", "Error parsing data " + e.toString());
        }
        return null;
    }
```

## P031

**A**

```java
public static void testAutoIncrement() {
        final int count = 3;
        final Object lock = new Object();
        for (int i = 0; i < count; i++) {
            new Thread(new Runnable() {

                @Override
                public void run() {
                    while (true) {
                        StringBuilder buffer = new StringBuilder(128);
                        buffer.append("insert into DOMAIN (                         ").append(LS);
                        buffer.append("    DOMAIN_ID, TOP_DOMAIN_ID, DOMAIN_HREF,   ").append(LS);
                        buffer.append("    DOMAIN_RANK, DOMAIN_TYPE, DOMAIN_STATUS, ").append(LS);
                        buffer.append("    DOMAIN_ICO_CREATED, DOMAIN_CDATE         ").append(LS);
                        buffer.append(") values (                   ").append(LS);
                        buffer.append("    null ,null, ?,").append(LS);
                        buffer.append("    1, 2, 1,                 ").append(LS);
                        buffer.append("    0, now()                 ").append(LS);
                        buffer.append(")                            ").append(LS);
                        String sqlInsert = buffer.toString();
                        boolean isAutoCommit = false;
                        int i = 0;
                        Connection conn = null;
                        PreparedStatement pstmt = null;
                        ResultSet rs = null;
                        try {
                            conn = ConnHelper.getConnection();
                            conn.setAutoCommit(isAutoCommit);
                            pstmt = conn.prepareStatement(sqlInsert);
                            for (i = 0; i < 10; i++) {
                                String lock = "" + ((int) (Math.random() * 100000000)) % 100;
                                pstmt.setString(1, lock);
                                pstmt.executeUpdate();
                            }
                            if (!isAutoCommit) conn.commit();
                            rs = pstmt.executeQuery("select max(DOMAIN_ID) from DOMAIN");
                            if (rs.next()) {
                                String str = System.currentTimeMillis() + " " + rs.getLong(1);
                            }
                        } catch (Exception e) {
                            try {
                                if (!isAutoCommit) conn.rollback();
                            } catch (SQLException ex) {
                                ex.printStackTrace(System.out);
                            }
                            String msg = System.currentTimeMillis() + " " + Thread.currentThread().getName() + " - " + i + " " + e.getMessage() + LS;
                            FileIO.writeToFile("D:/DEAD_LOCK.txt", msg, true, "GBK");
                        } finally {
                            ConnHelper.close(conn, pstmt, rs);
                        }
                    }
                }
            }).start();
        }
    }
```

**B**

```java
public void update() {
        new Thread(new Runnable() {

            @Override
            public void run() {
                try {
                    jButton1.setEnabled(false);
                    jButton2.setEnabled(false);
                    URL url = new URL(updatePath + "currentVersion.txt");
                    URLConnection con = url.openConnection();
                    con.connect();
                    BufferedReader in = new BufferedReader(new InputStreamReader(con.getInputStream()));
                    String line;
                    for (int i = 0; (line = in.readLine()) != null; i++) {
                        URL fileUrl = new URL(updatePath + line);
                        URLConnection filecon = fileUrl.openConnection();
                        InputStream stream = fileUrl.openStream();
                        int oneChar, count = 0;
                        int size = filecon.getContentLength();
                        jProgressBar1.setMaximum(size);
                        jProgressBar1.setValue(0);
                        File testFile = new File(line);
                        String build = "";
                        for (String dirtest : line.split("/")) {
                            build += dirtest;
                            if (!build.contains(".")) {
                                File dirfile = new File(build);
                                if (!dirfile.exists()) {
                                    dirfile.mkdir();
                                }
                            }
                            build += "/";
                        }
                        if (testFile.length() == size) {
                        } else {
                            transferFile(line, fileUrl, size);
                            if (line.endsWith("documents.zip")) {
                                ZipInputStream in2 = new ZipInputStream(new FileInputStream(line));
                                ZipEntry entry;
                                String pathDoc = line.split("documents.zip")[0];
                                File docDir = new File(pathDoc + "documents");
                                if (!docDir.exists()) {
                                    docDir.mkdir();
                                }
                                while ((entry = in2.getNextEntry()) != null) {
                                    String outFilename = pathDoc + "documents/" + entry.getName();
                                    OutputStream out = new BufferedOutputStream(new FileOutputStream(outFilename));
                                    byte[] buf = new byte[1024];
                                    int len;
                                    while ((len = in2.read(buf)) > 0) {
                                        out.write(buf, 0, len);
                                    }
                                    out.close();
                                }
                                in2.close();
                            }
                            if (line.endsWith("mysql.zip")) {
                                ZipFile zipfile = new ZipFile(line);
                                Enumeration entries = zipfile.entries();
                                String pathDoc = line.split("mysql.zip")[0];
                                File docDir = new File(pathDoc + "mysql");
                                if (!docDir.exists()) {
                                    docDir.mkdir();
                                }
                                while (entries.hasMoreElements()) {
                                    ZipEntry entry = (ZipEntry) entries.nextElement();
                                    if (entry.isDirectory()) {
                                        System.err.println("Extracting directory: " + entry.getName());
                                        (new File(pathDoc + "mysql/" + entry.getName())).mkdir();
                                        continue;
                                    }
                                    System.err.println("Extracting file: " + entry.getName());
                                    InputStream in2 = zipfile.getInputStream(entry);
                                    OutputStream out = new BufferedOutputStream(new FileOutputStream(pathDoc + "mysql/" + entry.getName()));
                                    byte[] buf = new byte[1024];
                                    int len;
                                    while ((len = in2.read(buf)) > 0) {
                                        out.write(buf, 0, len);
                                    }
                                    in2.close();
                                    out.close();
                                }
                            }
                        }
                        jProgressBar2.setValue(i + 1);
                        labelFileProgress.setText((i + 1) + "/" + numberFiles);
                    }
                    labelStatus.setText("Update Finished");
                    jButton1.setVisible(false);
                    jButton2.setText("Finished");
                    jButton1.setEnabled(true);
                    jButton2.setEnabled(true);
                } catch (IOException ex) {
                    Logger.getLogger(Updater.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }).start();
    }
```

## P032

**A**

```java
private static void copy(File source, File dest) throws FileNotFoundException, IOException {
        FileInputStream input = new FileInputStream(source);
        FileOutputStream output = new FileOutputStream(dest);
        System.out.println("Copying " + source + " to " + dest);
        IOUtils.copy(input, output);
        output.close();
        input.close();
        dest.setLastModified(source.lastModified());
    }
```

**B**

```java
protected void copyFile(File sourceFile, File destFile) {
        FileChannel in = null;
        FileChannel out = null;
        try {
            if (!verifyOrCreateParentPath(destFile.getParentFile())) {
                throw new IOException("Parent directory path " + destFile.getAbsolutePath() + " did not exist and could not be created");
            }
            if (destFile.exists() || destFile.createNewFile()) {
                in = new FileInputStream(sourceFile).getChannel();
                out = new FileOutputStream(destFile).getChannel();
                in.transferTo(0, in.size(), out);
            } else {
                throw new IOException("Couldn't create file for " + destFile.getAbsolutePath());
            }
        } catch (IOException ioe) {
            if (destFile.exists() && destFile.length() < sourceFile.length()) {
                destFile.delete();
            }
            ioe.printStackTrace();
        } finally {
            try {
                in.close();
            } catch (Throwable t) {
            }
            try {
                out.close();
            } catch (Throwable t) {
            }
            destFile.setLastModified(sourceFile.lastModified());
        }
    }
```

## P033

**A**

```java
private void copyFile(File sourceFile, File targetFile) {
        beNice();
        dispatchEvent(SynchronizationEventType.FileCopy, sourceFile, targetFile);
        File temporaryFile = new File(targetFile.getPath().concat(".jnstemp"));
        while (temporaryFile.exists()) {
            try {
                beNice();
                temporaryFile.delete();
                beNice();
            } catch (Exception ex) {
            }
        }
        try {
            if (targetFile.exists()) {
                targetFile.delete();
            }
            FileInputStream fis = new FileInputStream(sourceFile);
            FileOutputStream fos = new FileOutputStream(temporaryFile);
            byte[] buffer = new byte[204800];
            int readBytes = 0;
            int counter = 0;
            while ((readBytes = fis.read(buffer)) != -1) {
                counter++;
                updateStatus("... processing fragment " + String.valueOf(counter));
                fos.write(buffer, 0, readBytes);
            }
            fis.close();
            fos.close();
            temporaryFile.renameTo(targetFile);
            temporaryFile.setLastModified(sourceFile.lastModified());
            targetFile.setLastModified(sourceFile.lastModified());
        } catch (IOException e) {
            Exception dispatchedException = new Exception("ERROR: Copy File( " + sourceFile.getPath() + ", " + targetFile.getPath() + " )");
            dispatchEvent(dispatchedException, sourceFile, targetFile);
        }
        dispatchEvent(SynchronizationEventType.FileCopyDone, sourceFile, targetFile);
    }
```

**B**

```java
public static boolean copy(File source, File target, boolean owrite) {
        if (!source.exists()) {
            log.error("Invalid input to copy: source " + source + "doesn't exist");
            return false;
        } else if (!source.isFile()) {
            log.error("Invalid input to copy: source " + source + "isn't a file.");
            return false;
        } else if (target.exists() && !owrite) {
            log.error("Invalid input to copy: target " + target + " exists.");
            return false;
        }
        try {
            BufferedInputStream in = new BufferedInputStream(new FileInputStream(source));
            BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(target));
            byte buffer[] = new byte[1024];
            int read = -1;
            while ((read = in.read(buffer, 0, 1024)) != -1) out.write(buffer, 0, read);
            out.flush();
            out.close();
            in.close();
            return true;
        } catch (IOException e) {
            log.error("Copy failed: ", e);
            return false;
        }
    }
```

## P034

**A**

```java
protected void copyDependents() {
        for (File source : dependentFiles.keySet()) {
            try {
                if (!dependentFiles.get(source).exists()) {
                    if (dependentFiles.get(source).isDirectory()) dependentFiles.get(source).mkdirs(); else dependentFiles.get(source).getParentFile().mkdirs();
                }
                IOUtils.copyEverything(source, dependentFiles.get(source));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
```

**B**

```java
private static void loadManifests() {
        Perl5Util util = new Perl5Util();
        try {
            for (Enumeration e = classLoader.getResources("META-INF/MANIFEST.MF"); e.hasMoreElements(); ) {
                URL url = (URL) e.nextElement();
                if (util.match("/" + pluginPath.replace('\\', '/') + "/", url.getFile())) {
                    InputStream inputStream = url.openStream();
                    manifests.add(new Manifest(inputStream));
                    inputStream.close();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
```

## P035

**A**

```java
public static Checksum checksum(File file, Checksum checksum) throws IOException {
        if (file.isDirectory()) {
            throw new IllegalArgumentException("Checksums can't be computed on directories");
        }
        InputStream in = null;
        try {
            in = new CheckedInputStream(new FileInputStream(file), checksum);
            IOUtils.copy(in, new OutputStream() {

                @Override
                public void write(byte[] b, int off, int len) {
                }

                @Override
                public void write(int b) {
                }

                @Override
                public void write(byte[] b) throws IOException {
                }
            });
        } finally {
            IOUtils.closeQuietly(in);
        }
        return checksum;
    }
```

**B**

```java
public void calculate() throws FormatException, java.io.IOException {
        if (input == null) throw new IllegalStateException("FastaChecksummer input not set");
        contigHashes = new HashMap<String, ChecksumEntry>();
        String currentContig = null;
        java.security.MessageDigest hasher = null;
        try {
            hasher = java.security.MessageDigest.getInstance(checksumAlgorithm);
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new RuntimeException("Unexpected NoSuchAlgorithmException when asking for " + checksumAlgorithm + " algorithm");
        }
        String line = input.readLine();
        if (line == null) throw new FormatException("empty Fasta");
        try {
            while (line != null) {
                if (line.startsWith(">")) {
                    if (currentContig != null) {
                        String cs = new String(Hex.encodeHex(hasher.digest()));
                        contigHashes.put(currentContig, new ChecksumEntry(currentContig, cs));
                    }
                    Matcher m = ContigNamePattern.matcher(line);
                    if (m.matches()) {
                        currentContig = m.group(1);
                        hasher.reset();
                    } else throw new FormatException("Unexpected contig name format: " + line);
                } else {
                    if (currentContig == null) throw new FormatException("Sequence outside any fasta record (header is missing). Line: " + line); else hasher.update(line.getBytes("US-ASCII"));
                }
                line = input.readLine();
            }
            if (currentContig != null) {
                String cs = new String(Hex.encodeHex(hasher.digest()));
                contigHashes.put(currentContig, new ChecksumEntry(currentContig, cs));
            }
        } catch (java.io.UnsupportedEncodingException e) {
            throw new RuntimeException("Unexpected UnsupportedEncodingException! Line: " + line);
        }
    }
```

## P036

**A**

```java
public File addFile(File file, String suffix) throws IOException {
        if (file.exists() && file.isFile()) {
            File nf = File.createTempFile(prefix, "." + suffix, workdir);
            nf.delete();
            if (!file.renameTo(nf)) {
                IOUtils.copy(file, nf);
            }
            synchronized (fileList) {
                fileList.add(nf);
            }
            if (log.isDebugEnabled()) {
                log.debug("Add file [" + file.getPath() + "] -> [" + nf.getPath() + "]");
            }
            return nf;
        }
        return file;
    }
```

**B**

```java
public static void copyAssetFile(Context ctx, String srcFileName, String targetFilePath) {
        AssetManager assetManager = ctx.getAssets();
        try {
            InputStream is = assetManager.open(srcFileName);
            File out = new File(targetFilePath);
            if (!out.exists()) {
                out.getParentFile().mkdirs();
                out.createNewFile();
            }
            OutputStream os = new FileOutputStream(out);
            IOUtils.copy(is, os);
            is.close();
            os.close();
        } catch (IOException e) {
            AIOUtils.log("error when copyAssetFile", e);
        }
    }
```

## P037

**A**

```java
private void add(Hashtable applicantInfo) throws Exception {
        String mode = "".equals(getParam("applicant_id_gen").trim()) ? "update" : "insert";
        String applicant_id = getParam("applicant_id");
        String password = getParam("password");
        if ("".equals(applicant_id)) applicant_id = getParam("applicant_id_gen");
        if ("".equals(getParam("applicant_name"))) throw new Exception("Can not have empty fields!");
        applicantInfo.put("id", applicant_id);
        applicantInfo.put("password", password);
        applicantInfo.put("name", getParam("applicant_name"));
        applicantInfo.put("address1", getParam("address1"));
        applicantInfo.put("address2", getParam("address2"));
        applicantInfo.put("address3", getParam("address3"));
        applicantInfo.put("city", getParam("city"));
        applicantInfo.put("state", getParam("state"));
        applicantInfo.put("poscode", getParam("poscode"));
        applicantInfo.put("country_code", getParam("country_list"));
        applicantInfo.put("email", getParam("email"));
        applicantInfo.put("phone", getParam("phone"));
        String birth_year = getParam("birth_year");
        String birth_month = getParam("birth_month");
        String birth_day = getParam("birth_day");
        applicantInfo.put("birth_year", birth_year);
        applicantInfo.put("birth_month", birth_month);
        applicantInfo.put("birth_day", birth_day);
        applicantInfo.put("gender", getParam("gender"));
        String birth_date = birth_year + "-" + fmt(birth_month) + "-" + fmt(birth_day);
        applicantInfo.put("birth_date", birth_date);
        Db db = null;
        String sql = "";
        Connection conn = null;
        try {
            db = new Db();
            conn = db.getConnection();
            conn.setAutoCommit(false);
            Statement stmt = db.getStatement();
            SQLRenderer r = new SQLRenderer();
            boolean found = false;
            {
                r.add("applicant_id");
                r.add("applicant_id", (String) applicantInfo.get("id"));
                sql = r.getSQLSelect("adm_applicant");
                ResultSet rs = stmt.executeQuery(sql);
                if (rs.next()) found = true; else found = false;
            }
            if (found && !"update".equals(mode)) throw new Exception("Applicant Id was invalid!");
            {
                r.clear();
                r.add("password", (String) applicantInfo.get("password"));
                r.add("applicant_name", (String) applicantInfo.get("name"));
                r.add("address1", (String) applicantInfo.get("address1"));
                r.add("address2", (String) applicantInfo.get("address2"));
                r.add("address3", (String) applicantInfo.get("address3"));
                r.add("city", (String) applicantInfo.get("city"));
                r.add("state", (String) applicantInfo.get("state"));
                r.add("poscode", (String) applicantInfo.get("poscode"));
                r.add("country_code", (String) applicantInfo.get("country_code"));
                r.add("phone", (String) applicantInfo.get("phone"));
                r.add("birth_date", (String) applicantInfo.get("birth_date"));
                r.add("gender", (String) applicantInfo.get("gender"));
            }
            if (!found) {
                r.add("applicant_id", (String) applicantInfo.get("id"));
                sql = r.getSQLInsert("adm_applicant");
                stmt.executeUpdate(sql);
            } else {
                r.update("applicant_id", (String) applicantInfo.get("id"));
                sql = r.getSQLUpdate("adm_applicant");
                stmt.executeUpdate(sql);
            }
            conn.commit();
        } catch (DbException dbex) {
            throw dbex;
        } catch (SQLException sqlex) {
            try {
                conn.rollback();
            } catch (SQLException rollex) {
            }
            throw sqlex;
        } finally {
            if (db != null) db.close();
        }
    }
```

**B**

```java
public String addShare2(String appid, String appkey, String oauth_token, String oauth_token_secret, String openid, String format, Webpage webpage) throws Exception {
        String shareUrl = "http://openapi.qzone.qq.com/share/add_share";
        String oauth_signature = "";
        long oauth_timestamp = new Date().getTime() / 1000;
        String oauth_nonce = (Math.random() + "").replaceFirst("^0.", "");
        List<NameValuePair> shareParameters = new ArrayList<NameValuePair>();
        shareParameters.add(new BasicNameValuePair("format", format));
        shareParameters.add(new BasicNameValuePair("images", webpage.images));
        shareParameters.add(new BasicNameValuePair("oauth_consumer_key", appid));
        shareParameters.add(new BasicNameValuePair("oauth_nonce", oauth_nonce));
        shareParameters.add(new BasicNameValuePair("oauth_signature_method", "HMAC-SHA1"));
        shareParameters.add(new BasicNameValuePair("oauth_timestamp", oauth_timestamp + ""));
        shareParameters.add(new BasicNameValuePair("oauth_token", oauth_token));
        shareParameters.add(new BasicNameValuePair("oauth_version", "1.0"));
        shareParameters.add(new BasicNameValuePair("openid", openid));
        shareParameters.add(new BasicNameValuePair("title", webpage.title));
        shareParameters.add(new BasicNameValuePair("url", webpage.url));
        String stepA1 = "POST";
        String stepA2 = URLEncoder.encode(shareUrl, "UTF-8");
        String stepA3 = "";
        for (int i = 0; i < shareParameters.size(); i++) {
            NameValuePair item = shareParameters.get(i);
            stepA3 += item.getName() + "=" + item.getValue();
            if (i < shareParameters.size() - 1) {
                stepA3 += "&";
            }
        }
        stepA3 = URLEncoder.encode(stepA3, "UTF-8");
        String stepA = stepA1 + "&" + stepA2 + "&" + stepA3;
        String stepB = appkey + "&" + oauth_token_secret;
        Mac mac = Mac.getInstance("HmacSHA1");
        SecretKeySpec spec = new SecretKeySpec(stepB.getBytes("US-ASCII"), "HmacSHA1");
        mac.init(spec);
        byte[] oauthSignature = mac.doFinal(stepA.getBytes("US-ASCII"));
        oauth_signature = Base64Encoder.encode(oauthSignature);
        shareParameters.add(new BasicNameValuePair("oauth_signature", oauth_signature));
        HttpPost sharePost = new HttpPost(shareUrl);
        sharePost.setHeader("Referer", "http://openapi.qzone.qq.com");
        sharePost.setHeader("Host", "openapi.qzone.qq.com");
        sharePost.setHeader("Accept-Language", "zh-cn");
        sharePost.setHeader("Content-Type", "application/x-www-form-urlencoded");
        sharePost.setEntity(new UrlEncodedFormEntity(shareParameters, "UTF-8"));
        DefaultHttpClient httpclient = HttpClientUtils.getHttpClient();
        HttpResponse loginPostRes = httpclient.execute(sharePost);
        String shareHtml = HttpClientUtils.getHtml(loginPostRes, "UTF-8", false);
        return shareHtml;
    }
```

## P038

**A**

```java
public void testCodingCompletedFromFile() throws Exception {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        WritableByteChannel channel = newChannel(baos);
        HttpParams params = new BasicHttpParams();
        SessionOutputBuffer outbuf = new SessionOutputBufferImpl(1024, 128, params);
        HttpTransportMetricsImpl metrics = new HttpTransportMetricsImpl();
        LengthDelimitedEncoder encoder = new LengthDelimitedEncoder(channel, outbuf, metrics, 5);
        encoder.write(wrap("stuff"));
        File tmpFile = File.createTempFile("testFile", "txt");
        FileOutputStream fout = new FileOutputStream(tmpFile);
        OutputStreamWriter wrtout = new OutputStreamWriter(fout);
        wrtout.write("more stuff");
        wrtout.flush();
        wrtout.close();
        try {
            FileChannel fchannel = new FileInputStream(tmpFile).getChannel();
            encoder.transfer(fchannel, 0, 10);
            fail("IllegalStateException should have been thrown");
        } catch (IllegalStateException ex) {
        } finally {
            tmpFile.delete();
        }
    }
```

**B**

```java
public void testCodingFromFileSmaller() throws Exception {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        WritableByteChannel channel = newChannel(baos);
        HttpParams params = new BasicHttpParams();
        SessionOutputBuffer outbuf = new SessionOutputBufferImpl(1024, 128, params);
        HttpTransportMetricsImpl metrics = new HttpTransportMetricsImpl();
        LengthDelimitedEncoder encoder = new LengthDelimitedEncoder(channel, outbuf, metrics, 16);
        File tmpFile = File.createTempFile("testFile", "txt");
        FileOutputStream fout = new FileOutputStream(tmpFile);
        OutputStreamWriter wrtout = new OutputStreamWriter(fout);
        wrtout.write("stuff;");
        wrtout.write("more stuff;");
        wrtout.flush();
        wrtout.close();
        FileChannel fchannel = new FileInputStream(tmpFile).getChannel();
        encoder.transfer(fchannel, 0, 20);
        String s = baos.toString("US-ASCII");
        assertTrue(encoder.isCompleted());
        assertEquals("stuff;more stuff", s);
        tmpFile.delete();
    }
```

## P039

**A**

```java
public static synchronized String hash(String data) {
        if (digest == null) {
            try {
                digest = MessageDigest.getInstance("SHA-1");
            } catch (NoSuchAlgorithmException nsae) {
                System.err.println("Failed to load the SHA-1 MessageDigest. " + "Jive will be unable to function normally.");
            }
        }
        try {
            digest.update(data.getBytes("UTF-8"));
        } catch (UnsupportedEncodingException e) {
            System.err.println(e);
        }
        return encodeHex(digest.digest());
    }
```

**B**

```java
public static String encrypt(String plainText) throws Exception {
        MessageDigest md = null;
        try {
            md = MessageDigest.getInstance("MD5");
        } catch (NoSuchAlgorithmException e) {
            throw new Exception(e.getMessage());
        }
        try {
            md.update(plainText.getBytes("UTF-8"));
        } catch (UnsupportedEncodingException e) {
            throw new Exception(e.getMessage());
        }
        byte raw[] = md.digest();
        String hash = (new BASE64Encoder()).encode(raw);
        return hash;
    }
```

## P040

**A**

```java
public static String calcolaMd5(String messaggio) {
        MessageDigest md;
        try {
            md = MessageDigest.getInstance("MD5");
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
        md.reset();
        md.update(messaggio.getBytes());
        byte[] impronta = md.digest();
        return new String(impronta);
    }
```

**B**

```java
public static String encriptar(String string) throws Exception {
        MessageDigest md = null;
        try {
            md = MessageDigest.getInstance("MD5");
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
            throw new Exception("Algoritmo de Criptografia não encontrado.");
        }
        md.update(string.getBytes());
        BigInteger hash = new BigInteger(1, md.digest());
        String retorno = hash.toString(16);
        return retorno;
    }
```

## P041

**A**

```java
private void loadServers() {
        try {
            URL url = new URL(VirtualDeckConfig.SERVERS_URL);
            cmbServer.addItem("Local");
            BufferedReader in = new BufferedReader(new InputStreamReader(url.openStream()));
            String str;
            if (in.readLine().equals("[list]")) {
                while ((str = in.readLine()) != null) {
                    String[] host_line = str.split(";");
                    Host h = new Host();
                    h.setIp(host_line[0]);
                    h.setPort(Integer.parseInt(host_line[1]));
                    h.setName(host_line[2]);
                    getServers().add(h);
                    cmbServer.addItem(h.getName());
                }
            }
            in.close();
        } catch (MalformedURLException e) {
        } catch (IOException e) {
        }
    }
```

**B**

```java
int responseTomcat(InetAddress dest, int port, String request, boolean methodPost, StringBuffer response, int timeout) {
        int methodGetMaxSize = 250;
        int methodPostMaxSize = 32000;
        if (request == null || response == null) return -1;
        String fullRequest;
        if (methodPost) {
            String resource;
            String queryStr;
            int qIdx = request.indexOf('?');
            if (qIdx == -1) {
                resource = request;
                queryStr = "";
            } else {
                resource = request.substring(0, qIdx);
                queryStr = request.substring(qIdx + 1);
            }
            fullRequest = "POST " + resource + " HTTP/1.1\nHost: " + dest.getHostName() + ":" + (new Integer(port)).toString() + "\n\n" + queryStr;
        } else {
            fullRequest = "GET " + request + " HTTP/1.1\nHost: " + dest.getHostName() + ":" + (new Integer(port)).toString() + "\n\n";
        }
        if (methodPost && fullRequest.length() > methodPostMaxSize) {
            response.setLength(0);
            response.append("Complete POST request longer than maximum of " + methodPostMaxSize);
            return -5;
        } else if ((!methodPost) && fullRequest.length() > methodGetMaxSize) {
            response.setLength(0);
            response.append("Complete GET request longer than maximum of " + methodGetMaxSize);
            return -6;
        }
        String inputLine = "";
        request = "http://" + dest.getHostName() + ":" + (new Integer(port).toString()) + request;
        try {
            URL urlAddress = new URL(request);
            URLConnection urlC = urlAddress.openConnection();
            BufferedReader in = new BufferedReader(new InputStreamReader(urlC.getInputStream()));
            while ((inputLine = in.readLine()) != null) {
                response = response.append(inputLine).append("\n");
            }
        } catch (MalformedURLException e) {
            return -4;
        } catch (IOException e) {
            return -3;
        }
        return 200;
    }
```

## P042

**A**

```java
public Image storeImage(String title, String pathToImage, Map<String, Object> additionalProperties) {
        File collectionFolder = ProjectManager.getInstance().getFolder(PropertyHandler.getInstance().getProperty("_default_collection_name"));
        File imageFile = new File(pathToImage);
        String filename = "";
        String format = "";
        File copiedImageFile;
        while (true) {
            filename = "image" + UUID.randomUUID().hashCode();
            if (!DbEntryProvider.INSTANCE.idExists(filename)) {
                Path path = new Path(pathToImage);
                format = path.getFileExtension();
                copiedImageFile = new File(collectionFolder.getAbsolutePath() + File.separator + filename + "." + format);
                if (!copiedImageFile.exists()) break;
            }
        }
        try {
            copiedImageFile.createNewFile();
        } catch (IOException e1) {
            ExceptionHandlingService.INSTANCE.handleException(e1);
            return null;
        }
        BufferedInputStream in = null;
        BufferedOutputStream out = null;
        try {
            in = new BufferedInputStream(new FileInputStream(imageFile), 4096);
            out = new BufferedOutputStream(new FileOutputStream(copiedImageFile), 4096);
            int c;
            while ((c = in.read()) != -1) out.write(c);
            in.close();
            out.close();
        } catch (FileNotFoundException e) {
            ExceptionHandlingService.INSTANCE.handleException(e);
            return null;
        } catch (IOException e) {
            ExceptionHandlingService.INSTANCE.handleException(e);
            return null;
        }
        Image image = new ImageImpl();
        image.setId(filename);
        image.setFormat(format);
        image.setEntryDate(new Date());
        image.setTitle(title);
        image.setAdditionalProperties(additionalProperties);
        boolean success = DbEntryProvider.INSTANCE.storeNewImage(image);
        if (success) return image;
        return null;
    }
```

**B**

```java
public static void resize(File originalFile, File resizedFile, int width, String format) throws IOException {
        if (format != null && "gif".equals(format.toLowerCase())) {
            resize(originalFile, resizedFile, width, 1);
            return;
        }
        FileInputStream fis = new FileInputStream(originalFile);
        ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
        int readLength = -1;
        int bufferSize = 1024;
        byte bytes[] = new byte[bufferSize];
        while ((readLength = fis.read(bytes, 0, bufferSize)) != -1) {
            byteStream.write(bytes, 0, readLength);
        }
        byte[] in = byteStream.toByteArray();
        fis.close();
        byteStream.close();
        Image inputImage = Toolkit.getDefaultToolkit().createImage(in);
        waitForImage(inputImage);
        int imageWidth = inputImage.getWidth(null);
        if (imageWidth < 1) throw new IllegalArgumentException("image width " + imageWidth + " is out of range");
        int imageHeight = inputImage.getHeight(null);
        if (imageHeight < 1) throw new IllegalArgumentException("image height " + imageHeight + " is out of range");
        int height = -1;
        double scaleW = (double) imageWidth / (double) width;
        double scaleY = (double) imageHeight / (double) height;
        if (scaleW >= 0 && scaleY >= 0) {
            if (scaleW > scaleY) {
                height = -1;
            } else {
                width = -1;
            }
        }
        Image outputImage = inputImage.getScaledInstance(width, height, java.awt.Image.SCALE_DEFAULT);
        checkImage(outputImage);
        encode(new FileOutputStream(resizedFile), outputImage, format);
    }
```

## P043

**A**

```java
public MoteDeploymentConfiguration updateMoteDeploymentConfiguration(int mdConfigID, int programID, int radioPowerLevel) throws AdaptationException {
        MoteDeploymentConfiguration mdc = null;
        Connection connection = null;
        Statement statement = null;
        ResultSet resultSet = null;
        try {
            String query = "UPDATE MoteDeploymentConfigurations SET " + "programID       = " + programID + ", " + "radioPowerLevel = " + radioPowerLevel + "  " + "WHERE id = " + mdConfigID;
            connection = DriverManager.getConnection(CONN_STR);
            statement = connection.createStatement();
            statement.executeUpdate(query);
            query = "SELECT * from MoteDeploymentConfigurations WHERE " + "id = " + mdConfigID;
            resultSet = statement.executeQuery(query);
            if (!resultSet.next()) {
                connection.rollback();
                String msg = "Unable to select updated config.";
                log.error(msg);
                ;
                throw new AdaptationException(msg);
            }
            mdc = getMoteDeploymentConfiguration(resultSet);
            connection.commit();
        } catch (SQLException ex) {
            try {
                connection.rollback();
            } catch (Exception e) {
            }
            String msg = "SQLException in updateMoteDeploymentConfiguration";
            log.error(msg, ex);
            throw new AdaptationException(msg, ex);
        } finally {
            try {
                resultSet.close();
            } catch (Exception ex) {
            }
            try {
                statement.close();
            } catch (Exception ex) {
            }
            try {
                connection.close();
            } catch (Exception ex) {
            }
        }
        return mdc;
    }
```

**B**

```java
public int unindexRecord(String uuid) throws SQLException, CatalogIndexException {
        Connection con = null;
        boolean autoCommit = true;
        PreparedStatement st = null;
        int nRows = 0;
        StringSet fids = new StringSet();
        if (cswRemoteRepository.isActive()) {
            StringSet uuids = new StringSet();
            uuids.add(uuid);
            fids = queryFileIdentifiers(uuids);
        }
        try {
            con = returnConnection().getJdbcConnection();
            autoCommit = con.getAutoCommit();
            con.setAutoCommit(false);
            String sSql = "DELETE FROM " + getResourceDataTableName() + " WHERE DOCUUID=?";
            logExpression(sSql);
            st = con.prepareStatement(sSql);
            st.setString(1, uuid);
            nRows = st.executeUpdate();
            con.commit();
        } catch (SQLException ex) {
            if (con != null) {
                con.rollback();
            }
            throw ex;
        } finally {
            closeStatement(st);
            if (con != null) {
                con.setAutoCommit(autoCommit);
            }
        }
        CatalogIndexAdapter indexAdapter = getCatalogIndexAdapter();
        if (indexAdapter != null) {
            indexAdapter.deleteDocument(uuid);
            if (cswRemoteRepository.isActive()) {
                if (fids.size() > 0) cswRemoteRepository.onRecordsDeleted(fids);
            }
        }
        return nRows;
    }
```

## P044

**A**

```java
private void fileCopy(final File src, final File dest) throws IOException {
        final FileChannel srcChannel = new FileInputStream(src).getChannel();
        final FileChannel dstChannel = new FileOutputStream(dest).getChannel();
        dstChannel.transferFrom(srcChannel, 0, srcChannel.size());
        srcChannel.close();
        dstChannel.close();
    }
```

**B**

```java
private void channelCopy(File source, File dest) throws IOException {
        FileChannel srcChannel = new FileInputStream(source).getChannel();
        FileChannel dstChannel = new FileOutputStream(dest).getChannel();
        try {
            dstChannel.transferFrom(srcChannel, 0, srcChannel.size());
        } finally {
            srcChannel.close();
            dstChannel.close();
        }
    }
```

## P045

**A**

```java
public static boolean copy(File source, File target) {
        try {
            if (!source.exists()) return false;
            target.getParentFile().mkdirs();
            InputStream input = new FileInputStream(source);
            OutputStream output = new FileOutputStream(target);
            byte[] buf = new byte[1024];
            int len;
            while ((len = input.read(buf)) > 0) output.write(buf, 0, len);
            input.close();
            output.close();
            return true;
        } catch (Exception exc) {
            exc.printStackTrace();
            return false;
        }
    }
```

**B**

```java
private void resourceCopy(String resource, IProject project, String target, IProgressMonitor monitor, Map<String, String> replacement, String charset) throws URISyntaxException, IOException {
        IFile targetFile = project.getFile(target);
        URL url = bundle.getEntry(resource);
        InputStream is = null;
        ByteArrayInputStream bais = null;
        try {
            is = FileLocator.toFileURL(url).openStream();
            int len = is.available();
            byte[] buf = new byte[len];
            is.read(buf);
            String str = new String(buf, charset);
            for (String toRepl : replacement.keySet()) {
                str = str.replaceAll(toRepl, replacement.get(toRepl));
            }
            bais = new ByteArrayInputStream(str.getBytes("UTF-8"));
            if (targetFile.exists()) {
                targetFile.setContents(bais, true, false, monitor);
            } else {
                targetFile.create(bais, true, monitor);
            }
        } catch (Exception e) {
            throw new IOException(e);
        } finally {
            if (bais != null) {
                bais.close();
            }
            if (is != null) {
                is.close();
            }
        }
    }
```

## P046

**A**

```java
public void copyContent(long mailId1, long mailId2) throws Exception {
        File file1 = new File(this.getMailDir(mailId1) + "/");
        File file2 = new File(this.getMailDir(mailId2) + "/");
        this.recursiveDir(file2);
        if (file1.isDirectory()) {
            File[] files = file1.listFiles();
            if (files != null) {
                for (int i = 0; i < files.length; i++) {
                    if (files[i].isFile()) {
                        File file2s = new File(file2.getAbsolutePath() + "/" + files[i].getName());
                        if (!file2s.exists()) {
                            file2s.createNewFile();
                            BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(file2s));
                            BufferedInputStream in = new BufferedInputStream(new FileInputStream(files[i]));
                            int read;
                            while ((read = in.read()) != -1) {
                                out.write(read);
                            }
                            out.flush();
                            if (in != null) {
                                try {
                                    in.close();
                                } catch (IOException ex1) {
                                    ex1.printStackTrace();
                                }
                            }
                            if (out != null) {
                                try {
                                    out.close();
                                } catch (IOException ex) {
                                    ex.printStackTrace();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
```

**B**

```java
public static boolean makeBackup(File dir, String sourcedir, String destinationdir, String destinationDirEnding, boolean autoInitialized) {
        boolean success = false;
        String[] files;
        files = dir.list();
        File checkdir = new File(destinationdir + System.getProperty("file.separator") + destinationDirEnding);
        if (!checkdir.isDirectory()) {
            checkdir.mkdir();
        }
        ;
        Date date = new Date();
        long msec = date.getTime();
        checkdir.setLastModified(msec);
        try {
            for (int i = 0; i < files.length; i++) {
                File f = new File(dir, files[i]);
                File g = new File(files[i]);
                if (f.isDirectory()) {
                } else if (f.getName().endsWith("saving")) {
                } else {
                    if (f.canRead()) {
                        String destinationFile = checkdir + System.getProperty("file.separator") + g;
                        String sourceFile = sourcedir + System.getProperty("file.separator") + g;
                        FileInputStream infile = new FileInputStream(sourceFile);
                        FileOutputStream outfile = new FileOutputStream(destinationFile);
                        int c;
                        while ((c = infile.read()) != -1) outfile.write(c);
                        infile.close();
                        outfile.close();
                    } else {
                        System.out.println(f.getName() + " is LOCKED!");
                        while (!f.canRead()) {
                        }
                        String destinationFile = checkdir + System.getProperty("file.separator") + g;
                        String sourceFile = sourcedir + System.getProperty("file.separator") + g;
                        FileInputStream infile = new FileInputStream(sourceFile);
                        FileOutputStream outfile = new FileOutputStream(destinationFile);
                        int c;
                        while ((c = infile.read()) != -1) outfile.write(c);
                        infile.close();
                        outfile.close();
                    }
                }
            }
            success = true;
        } catch (Exception e) {
            success = false;
            e.printStackTrace();
        }
        if (autoInitialized) {
            Display display = View.getDisplay();
            if (display != null || !display.isDisposed()) {
                View.getDisplay().syncExec(new Runnable() {

                    public void run() {
                        Tab4.redrawBackupTable();
                    }
                });
            }
            return success;
        } else {
            View.getDisplay().syncExec(new Runnable() {

                public void run() {
                    StatusBoxUtils.mainStatusAdd(" Backup Complete", 1);
                    View.getPluginInterface().getPluginconfig().setPluginParameter("Azcvsupdater_last_backup", Time.getCurrentTime(View.getPluginInterface().getPluginconfig().getPluginBooleanParameter("MilitaryTime")));
                    Tab4.lastBackupTime = View.getPluginInterface().getPluginconfig().getPluginStringParameter("Azcvsupdater_last_backup");
                    if (Tab4.lastbackupValue != null || !Tab4.lastbackupValue.isDisposed()) {
                        Tab4.lastbackupValue.setText("Last backup: " + Tab4.lastBackupTime);
                    }
                    Tab4.redrawBackupTable();
                    Tab6Utils.refreshLists();
                }
            });
            return success;
        }
    }
```

## P047

**A**

```java
public static void main(String[] args) throws IOException {
        MSPack pack = new MSPack(new FileInputStream(args[0]));
        String[] files = pack.getFileNames();
        for (int i = 0; i < files.length; i++) System.out.println(i + ": " + files[i] + ": " + pack.getLengths()[i]);
        System.out.println("Writing " + files[files.length - 1]);
        InputStream is = pack.getInputStream(files.length - 1);
        OutputStream os = new FileOutputStream(files[files.length - 1]);
        int n;
        byte[] buf = new byte[4096];
        while ((n = is.read(buf)) != -1) os.write(buf, 0, n);
        os.close();
        is.close();
    }
```

**B**

```java
public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            System.out.println("Usage: \nGZIPcompress file\n" + "\tUses GZIP compression to compress " + "the file to test.gz");
            System.exit(1);
        }
        BufferedReader in = new BufferedReader(new FileReader(args[0]));
        BufferedOutputStream out = new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream("test.gz")));
        System.out.println("Writing file");
        int c;
        while ((c = in.read()) != -1) out.write(c);
        in.close();
        out.close();
        System.out.println("Reading file");
        BufferedReader in2 = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream("test.gz"))));
        String s;
        while ((s = in2.readLine()) != null) System.out.println(s);
    }
```

## P048

**A**

```java
public static void generateCode(File flowFile, String packagePath, File destDir, File scriptRootFolder) throws IOException {
        InputStream javaSrcIn = generateCode(flowFile, packagePath, scriptRootFolder);
        File outputFolder = new File(destDir, packagePath.replace('.', File.separatorChar));
        String fileName = flowFile.getName();
        fileName = fileName.substring(0, fileName.lastIndexOf(".") + 1) + Consts.FILE_EXTENSION_GROOVY;
        File outputFile = new File(outputFolder, fileName);
        OutputStream javaSrcOut = new FileOutputStream(outputFile);
        IOUtils.copyBufferedStream(javaSrcIn, javaSrcOut);
        javaSrcIn.close();
        javaSrcOut.close();
    }
```

**B**

```java
private String sendMessage(HttpURLConnection connection, String reqMessage) throws IOException, XMLStreamException {
        if (msgLog.isTraceEnabled()) msgLog.trace("Outgoing SOAPMessage\n" + reqMessage);
        BufferedOutputStream out = new BufferedOutputStream(connection.getOutputStream());
        out.write(reqMessage.getBytes("UTF-8"));
        out.close();
        InputStream inputStream = null;
        if (connection.getResponseCode() < 400) inputStream = connection.getInputStream(); else inputStream = connection.getErrorStream();
        ByteArrayOutputStream baos = new ByteArrayOutputStream(1024);
        IOUtils.copyStream(baos, inputStream);
        inputStream.close();
        byte[] byteArray = baos.toByteArray();
        String resMessage = new String(byteArray, "UTF-8");
        if (msgLog.isTraceEnabled()) msgLog.trace("Incoming Response SOAPMessage\n" + resMessage);
        return resMessage;
    }
```

## P049

**A**

```java
public static void copyFile(File in, File out) {
        try {
            FileChannel sourceChannel = new FileInputStream(in).getChannel();
            FileChannel destinationChannel = new FileOutputStream(out).getChannel();
            sourceChannel.transferTo(0, sourceChannel.size(), destinationChannel);
            sourceChannel.close();
            destinationChannel.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
```

**B**

```java
public static void copyFile(File in, File out) throws EnhancedException {
        try {
            FileChannel sourceChannel = new FileInputStream(in).getChannel();
            FileChannel destinationChannel = new FileOutputStream(out).getChannel();
            sourceChannel.transferTo(0, sourceChannel.size(), destinationChannel);
            sourceChannel.close();
            destinationChannel.close();
        } catch (Exception e) {
            throw new EnhancedException("Could not copy file " + in.getAbsolutePath() + " to " + out.getAbsolutePath() + ".", e);
        }
    }
```

## P050

**A**

```java
private static String doGetForSessionKey(String authCode) throws Exception {
        String sessionKey = "";
        HttpClient hc = new DefaultHttpClient();
        HttpGet hg = new HttpGet(Common.TEST_SESSION_HOST + Common.TEST_SESSION_PARAM + authCode);
        HttpResponse hr = hc.execute(hg);
        BufferedReader br = new BufferedReader(new InputStreamReader(hr.getEntity().getContent()));
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = br.readLine()) != null) {
            sb.append(line);
        }
        String result = sb.toString();
        Log.i("sessionKeyMessages", result);
        Map<String, String> map = Util.handleURLParameters(result);
        sessionKey = map.get(Common.TOP_SESSION);
        String topParameters = map.get(Common.TOP_PARAMETERS);
        String decTopParameters = Util.decodeBase64(topParameters);
        Log.i("base64", decTopParameters);
        map = Util.handleURLParameters(decTopParameters);
        Log.i("nick", map.get(Common.VISITOR_NICK));
        CachePool.put(Common.VISITOR_NICK, map.get(Common.VISITOR_NICK));
        return sessionKey;
    }
```

**B**

```java
public byte[] evaluateResponse(byte[] responseBytes) throws SaslException {
        if (firstEvaluation) {
            firstEvaluation = false;
            StringBuilder challenge = new StringBuilder(100);
            Iterator iter = configurationManager.getRealms().values().iterator();
            Realm aRealm;
            while (iter.hasNext()) {
                aRealm = (Realm) iter.next();
                if (aRealm.getFullRealmName().equals("null")) continue;
                challenge.append("realm=\"" + aRealm.getFullRealmName() + "\"");
                challenge.append(",");
            }
            String nonceUUID = UUID.randomUUID().toString();
            String nonce = null;
            try {
                nonce = new String(Base64.encodeBase64(MD5Digest(String.valueOf(System.nanoTime() + ":" + nonceUUID))), "US-ASCII");
            } catch (UnsupportedEncodingException uee) {
                throw new SaslException(uee.getMessage(), uee);
            } catch (GeneralSecurityException uee) {
                throw new SaslException(uee.getMessage(), uee);
            }
            nonces.put(nonce, new ArrayList());
            nonces.get(nonce).add(Integer.valueOf(1));
            challenge.append("nonce=\"" + nonce + "\"");
            challenge.append(",");
            challenge.append("qop=\"" + configurationManager.getSaslQOP() + "\"");
            challenge.append(",");
            challenge.append("charset=\"utf-8\"");
            challenge.append(",");
            challenge.append("algorithm=\"md5-sess\"");
            if (configurationManager.getSaslQOP().indexOf("auth-conf") != -1) {
                challenge.append(",");
                challenge.append("cipher-opts=\"" + configurationManager.getDigestMD5Ciphers() + "\"");
            }
            try {
                return Base64.encodeBase64(challenge.toString().getBytes("US-ASCII"));
            } catch (UnsupportedEncodingException uee) {
                throw new SaslException(uee.getMessage(), uee);
            }
        } else {
            String nonce = null;
            if (!Base64.isArrayByteBase64(responseBytes)) {
                throw new SaslException("Can not decode Base64 Content", new MalformedBase64ContentException());
            }
            responseBytes = Base64.decodeBase64(responseBytes);
            List<byte[]> splittedBytes = splitByteArray(responseBytes, (byte) 0x3d);
            int tokenCountMinus1 = splittedBytes.size() - 1, lastCommaPos;
            Map rawDirectives = new HashMap();
            String key = null;
            Map<String, String> directives;
            try {
                key = new String(splittedBytes.get(0), "US-ASCII");
                for (int i = 1; i < tokenCountMinus1; i++) {
                    key = responseTokenProcessor(splittedBytes, rawDirectives, key, i, tokenCountMinus1);
                }
                responseTokenProcessor(splittedBytes, rawDirectives, key, tokenCountMinus1, tokenCountMinus1);
                if (rawDirectives.containsKey("charset")) {
                    String value = new String((byte[]) rawDirectives.get("charset"), "US-ASCII").toLowerCase(locale);
                    if (value.equals("utf-8")) {
                        encoding = "UTF-8";
                    }
                }
                if (encoding.equals("ISO-8859-1")) {
                    decodeAllAs8859(rawDirectives);
                } else {
                    decodeMixed(rawDirectives);
                }
                directives = rawDirectives;
            } catch (UnsupportedEncodingException uee) {
                throw new SaslException(uee.getMessage());
            }
            if (!directives.containsKey("username") || !directives.containsKey("nonce") || !directives.containsKey("nc") || !directives.containsKey("cnonce") || !directives.containsKey("response")) {
                throw new SaslException("Digest-Response lacks at least one neccesery key-value pair");
            }
            if (directives.get("username").indexOf('@') != -1) {
                throw new SaslException("digest-response username field must not include domain name", new AuthenticationException());
            }
            if (!directives.containsKey("qop")) {
                directives.put("qop", QOP_AUTH);
            }
            if (!directives.containsKey("realm") || ((String) directives.get("realm")).equals("")) {
                directives.put("realm", "null");
            }
            nonce = (String) directives.get("nonce");
            if (!nonces.containsKey(nonce)) {
                throw new SaslException("Illegal nonce value");
            }
            List<Integer> nonceListInMap = nonces.get(nonce);
            int nc = Integer.parseInt((String) directives.get("nc"), 16);
            if (nonceListInMap.get(nonceListInMap.size() - 1).equals(Integer.valueOf(nc))) {
                nonceListInMap.add(Integer.valueOf(++nc));
            } else {
                throw new SaslException("Illegal nc value");
            }
            nonceListInMap = null;
            if (directives.get("qop").equals(QOP_AUTH_INT)) integrity = true; else if (directives.get("qop").equals(QOP_AUTH_CONF)) privacy = true;
            if (privacy) {
                if (!directives.containsKey("cipher")) {
                    throw new SaslException("Message confidentially required but cipher entry is missing");
                }
                sessionCipher = directives.get("cipher").toLowerCase(locale);
                if ("3des,des,rc4-40,rc4,rc4-56".indexOf(sessionCipher) == -1) {
                    throw new SaslException("Unsupported cipher for message confidentiality");
                }
            }
            String realm = directives.get("realm").toLowerCase(Locale.getDefault());
            String username = directives.get("username").toLowerCase(locale);
            if (username.indexOf('@') == -1) {
                if (!directives.get("realm").equals("null")) {
                    username += directives.get("realm").substring(directives.get("realm").indexOf('@'));
                } else if (directives.get("authzid").indexOf('@') != -1) {
                    username += directives.get("authzid").substring(directives.get("authzid").indexOf('@'));
                }
            }
            DomainWithPassword domainWithPassword = configurationManager.getRealmPassword(realm, username);
            if (domainWithPassword == null || domainWithPassword.getPassword() == null) {
                log.warn("The supplied username and/or realm do(es) not match a registered entry");
                return null;
            }
            if (realm.equals("null") && username.indexOf('@') == -1) {
                username += "@" + domainWithPassword.getDomain();
            }
            byte[] HA1 = toByteArray(domainWithPassword.getPassword());
            for (int i = domainWithPassword.getPassword().length - 1; i >= 0; i--) {
                domainWithPassword.getPassword()[i] = 0xff;
            }
            domainWithPassword = null;
            MessageDigest md = null;
            try {
                md = MessageDigest.getInstance("MD5");
            } catch (GeneralSecurityException gse) {
                throw new SaslException(gse.getMessage());
            }
            md.update(HA1);
            md.update(":".getBytes());
            md.update((directives.get("nonce")).getBytes());
            md.update(":".getBytes());
            md.update((directives.get("cnonce")).getBytes());
            if (directives.containsKey("authzid")) {
                md.update(":".getBytes());
                md.update((directives.get("authzid")).getBytes());
            }
            MD5DigestSessionKey = HA1 = md.digest();
            String MD5DigestSessionKeyToHex = toHex(HA1, HA1.length);
            md.update("AUTHENTICATE".getBytes());
            md.update(":".getBytes());
            md.update((directives.get("digest-uri")).getBytes());
            if (!directives.get("qop").equals(QOP_AUTH)) {
                md.update(":".getBytes());
                md.update("00000000000000000000000000000000".getBytes());
            }
            byte[] HA2 = md.digest();
            String HA2HEX = toHex(HA2, HA2.length);
            md.update(MD5DigestSessionKeyToHex.getBytes());
            md.update(":".getBytes());
            md.update((directives.get("nonce")).getBytes());
            md.update(":".getBytes());
            md.update((directives.get("nc")).getBytes());
            md.update(":".getBytes());
            md.update((directives.get("cnonce")).getBytes());
            md.update(":".getBytes());
            md.update((directives.get("qop")).getBytes());
            md.update(":".getBytes());
            md.update(HA2HEX.getBytes());
            byte[] responseHash = md.digest();
            String HexResponseHash = toHex(responseHash, responseHash.length);
            if (HexResponseHash.equals(directives.get("response"))) {
                md.update(":".getBytes());
                md.update((directives.get("digest-uri")).getBytes());
                if (!directives.get("qop").equals(QOP_AUTH)) {
                    md.update(":".getBytes());
                    md.update("00000000000000000000000000000000".getBytes());
                }
                HA2 = md.digest();
                HA2HEX = toHex(HA2, HA2.length);
                md.update(MD5DigestSessionKeyToHex.getBytes());
                md.update(":".getBytes());
                md.update((directives.get("nonce")).getBytes());
                md.update(":".getBytes());
                md.update((directives.get("nc")).getBytes());
                md.update(":".getBytes());
                md.update((directives.get("cnonce")).getBytes());
                md.update(":".getBytes());
                md.update((directives.get("qop")).getBytes());
                md.update(":".getBytes());
                md.update(HA2HEX.getBytes());
                responseHash = md.digest();
                return finalizeAuthentication.finalize(responseHash, username);
            } else {
                log.warn("Improper credentials");
                return null;
            }
        }
    }
```

## P051

**A**

```java
public ChatClient registerPlayer(int playerId, String playerLogin) throws NoSuchAlgorithmException, UnsupportedEncodingException {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        md.reset();
        md.update(playerLogin.getBytes("UTF-8"), 0, playerLogin.length());
        byte[] accountToken = md.digest();
        byte[] token = generateToken(accountToken);
        ChatClient chatClient = new ChatClient(playerId, token);
        players.put(playerId, chatClient);
        return chatClient;
    }
```

**B**

```java
private String getAuthCookie(boolean invalidate) {
        if (resources.getBoolean(R.bool.dev)) {
            return "dev_appserver_login=get_view@localhost.devel:false:18580476422013912411";
        } else {
            try {
                Account[] accounts = accountsManager.getAccountsByType("com.google");
                Account account = null;
                while (!(accounts.length > 0)) {
                    accountsManager.addAccount("com.google", "ah", null, null, act, null, null).getResult();
                    accounts = accountsManager.getAccountsByType("com.google");
                }
                if (account == null) {
                    account = accounts[0];
                }
                String authToken = accountsManager.getAuthToken(account, "ah", null, act, null, null).getResult().get(AccountManager.KEY_AUTHTOKEN).toString();
                if (invalidate || authToken == null) {
                    Logger.getLogger(JSBridge.class.getName()).log(Level.INFO, "Invalidating auth token.");
                    accountsManager.invalidateAuthToken("com.google", authToken);
                    return getAuthCookie(false);
                }
                HttpGet httpget = new HttpGet("http://" + resources.getString(R.string.host) + "/_ah/login?auth=" + authToken);
                HttpResponse response = httpclient.execute(httpget);
                for (Header c : response.getHeaders("Set-Cookie")) {
                    if (c.getValue().startsWith("ACSID=")) {
                        return c.getValue();
                    }
                }
                return getAuthCookie(false);
            } catch (ClientProtocolException e) {
                Logger.getLogger(JSBridge.class.getName()).log(Level.SEVERE, "HTTP protocol violated.", e);
            } catch (OperationCanceledException e) {
                Logger.getLogger(JSBridge.class.getName()).log(Level.WARNING, "Login canceled.", e);
            } catch (AuthenticatorException e) {
                Logger.getLogger(JSBridge.class.getName()).log(Level.WARNING, "Authentication failed.", e);
            } catch (IOException e) {
                Logger.getLogger(JSBridge.class.getName()).log(Level.SEVERE, "Login failed.", e);
            }
            return getAuthCookie(true);
        }
    }
```

## P052

**A**

```java
public static synchronized String toMD5(String str) {
        Nulls.failIfNull(str, "Cannot create an MD5 encryption form a NULL string");
        String hashword = null;
        try {
            MessageDigest md5 = MessageDigest.getInstance(MD5);
            md5.update(str.getBytes());
            BigInteger hash = new BigInteger(1, md5.digest());
            hashword = hash.toString(16);
            return Strings.padLeft(hashword, 32, "0");
        } catch (NoSuchAlgorithmException ex) {
            ex.printStackTrace();
        }
        return hashword;
    }
```

**B**

```java
public static String getPasswordHash(String password) {
        MessageDigest md;
        try {
            md = MessageDigest.getInstance("MD5");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalArgumentException(e);
        }
        md.update(password.getBytes());
        byte[] digest = md.digest();
        BigInteger i = new BigInteger(1, digest);
        String hash = i.toString(16);
        while (hash.length() < 32) {
            hash = "0" + hash;
        }
        return hash;
    }
```

## P053

**A**

```java
public void putFile(CompoundName file, FileInputStream fileInput) throws IOException {
        File fullDir = new File(REMOTE_BASE_DIR.getCanonicalPath());
        for (int i = 0; i < file.size() - 1; i++) fullDir = new File(fullDir, file.get(i));
        fullDir.mkdirs();
        File outputFile = new File(fullDir, file.get(file.size() - 1));
        FileOutputStream outStream = new FileOutputStream(outputFile);
        for (int byteIn = fileInput.read(); byteIn != -1; byteIn = fileInput.read()) outStream.write(byteIn);
        fileInput.close();
        outStream.close();
    }
```

**B**

```java
private void initFiles() throws IOException {
        if (!tempDir.exists()) {
            if (!tempDir.mkdir()) throw new IOException("Temp dir '' can not be created");
        }
        File tmp = new File(tempDir, TORRENT_FILENAME);
        if (!tmp.exists()) {
            FileChannel in = new FileInputStream(torrentFile).getChannel();
            FileChannel out = new FileOutputStream(tmp).getChannel();
            in.transferTo(0, in.size(), out);
            in.close();
            out.close();
        }
        torrentFile = tmp;
        if (!stateFile.exists()) {
            FileChannel out = new FileOutputStream(stateFile).getChannel();
            int numChunks = metadata.getPieceHashes().size();
            ByteBuffer zero = ByteBuffer.wrap(new byte[] { 0, 0, 0, 0 });
            for (int i = 0; i < numChunks; i++) {
                out.write(zero);
                zero.clear();
            }
            out.close();
        }
    }
```

## P054

**A**

```java
public static void main(String[] a) {
        ArrayList<String> allFilesToBeCopied = new ArrayList<String>();
        new File(outputDir).mkdirs();
        try {
            FileReader fis = new FileReader(completeFileWithDirToCathFileList);
            BufferedReader bis = new BufferedReader(fis);
            String line = "";
            String currentCombo = "";
            while ((line = bis.readLine()) != null) {
                String[] allEntries = line.split("\\s+");
                String fileName = allEntries[0];
                String thisCombo = allEntries[1] + allEntries[2] + allEntries[3] + allEntries[4];
                if (currentCombo.equals(thisCombo)) {
                } else {
                    System.out.println("merke: " + fileName);
                    allFilesToBeCopied.add(fileName);
                    currentCombo = thisCombo;
                }
            }
            System.out.println(allFilesToBeCopied.size());
            for (String file : allFilesToBeCopied) {
                try {
                    FileChannel srcChannel = new FileInputStream(CathDir + file).getChannel();
                    FileChannel dstChannel = new FileOutputStream(outputDir + file).getChannel();
                    dstChannel.transferFrom(srcChannel, 0, srcChannel.size());
                    srcChannel.close();
                    dstChannel.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
```

**B**

```java
public void sorter() {
        String inputLine1, inputLine2;
        String epiNames[] = new String[1000];
        String epiEpisodes[] = new String[1000];
        int lineCounter = 0;
        try {
            String pluginDir = pluginInterface.getPluginDirectoryName();
            String eplist_file = pluginDir + System.getProperty("file.separator") + "EpisodeList.txt";
            File episodeList = new File(eplist_file);
            if (!episodeList.isFile()) {
                episodeList.createNewFile();
            }
            final BufferedReader in = new BufferedReader(new FileReader(episodeList));
            while ((inputLine1 = in.readLine()) != null) {
                if ((inputLine2 = in.readLine()) != null) {
                    epiNames[lineCounter] = inputLine1;
                    epiEpisodes[lineCounter] = inputLine2;
                    lineCounter++;
                }
            }
            in.close();
            int epiLength = epiNames.length;
            for (int i = 0; i < (lineCounter); i++) {
                for (int j = 0; j < (lineCounter - 1); j++) {
                    if (epiNames[j].compareToIgnoreCase(epiNames[j + 1]) > 0) {
                        String temp = epiNames[j];
                        epiNames[j] = epiNames[j + 1];
                        epiNames[j + 1] = temp;
                        String temp2 = epiEpisodes[j];
                        epiEpisodes[j] = epiEpisodes[j + 1];
                        epiEpisodes[j + 1] = temp2;
                    }
                }
            }
            File episodeList2 = new File(eplist_file);
            BufferedWriter bufWriter = new BufferedWriter(new FileWriter(episodeList2));
            for (int i = 0; i <= lineCounter; i++) {
                if (epiNames[i] == null) {
                    break;
                }
                bufWriter.write(epiNames[i] + "\n");
                bufWriter.write(epiEpisodes[i] + "\n");
            }
            bufWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
```

## P055

**A**

```java
public static void copy(String srcFileName, String destFileName) throws IOException {
        if (srcFileName == null) {
            throw new IllegalArgumentException("srcFileName is null");
        }
        if (destFileName == null) {
            throw new IllegalArgumentException("destFileName is null");
        }
        FileChannel src = null;
        FileChannel dest = null;
        try {
            src = new FileInputStream(srcFileName).getChannel();
            dest = new FileOutputStream(destFileName).getChannel();
            long n = src.size();
            MappedByteBuffer buf = src.map(FileChannel.MapMode.READ_ONLY, 0, n);
            dest.write(buf);
        } finally {
            if (dest != null) {
                try {
                    dest.close();
                } catch (IOException e1) {
                }
            }
            if (src != null) {
                try {
                    src.close();
                } catch (IOException e1) {
                }
            }
        }
    }
```

**B**

```java
public static boolean copyFile(File sourceFile, File destFile) {
        FileChannel srcChannel = null;
        FileChannel dstChannel = null;
        try {
            srcChannel = new FileInputStream(sourceFile).getChannel();
            dstChannel = new FileOutputStream(destFile).getChannel();
            long pos = 0;
            long count = srcChannel.size();
            if (count > MAX_BLOCK_SIZE) {
                count = MAX_BLOCK_SIZE;
            }
            long transferred = Long.MAX_VALUE;
            while (transferred > 0) {
                transferred = dstChannel.transferFrom(srcChannel, pos, count);
                pos = transferred;
            }
        } catch (IOException e) {
            return false;
        } finally {
            if (srcChannel != null) {
                try {
                    srcChannel.close();
                } catch (IOException e) {
                }
            }
            if (dstChannel != null) {
                try {
                    dstChannel.close();
                } catch (IOException e) {
                }
            }
        }
        return true;
    }
```

## P056

**A**

```java
private void getRandomGUID(boolean secure) throws NoSuchAlgorithmException {
        MessageDigest md5 = null;
        StringBuffer sbValueBeforeMD5 = new StringBuffer();
        try {
            md5 = MessageDigest.getInstance("MD5");
        } catch (NoSuchAlgorithmException e) {
            System.out.println("Error: " + e);
            throw e;
        }
        try {
            long time = System.currentTimeMillis();
            long rand = 0;
            if (secure) {
                rand = mySecureRand.nextLong();
            } else {
                rand = myRand.nextLong();
            }
            sbValueBeforeMD5.append(s_id);
            sbValueBeforeMD5.append(":");
            sbValueBeforeMD5.append(Long.toString(time));
            sbValueBeforeMD5.append(":");
            sbValueBeforeMD5.append(Long.toString(rand));
            valueBeforeMD5 = sbValueBeforeMD5.toString();
            md5.update(valueBeforeMD5.getBytes());
            byte[] array = md5.digest();
            StringBuffer sb = new StringBuffer();
            for (int j = 0; j < array.length; ++j) {
                int b = array[j] & 0xFF;
                if (b < 0x10) sb.append('0');
                sb.append(Integer.toHexString(b));
            }
            valueAfterMD5 = sb.toString();
        } catch (Exception e) {
            System.out.println("Error:" + e);
        }
    }
```

**B**

```java
private void getRandomGUID(boolean secure) {
        MessageDigest md5 = null;
        StringBuffer sbValueBeforeMD5 = new StringBuffer();
        try {
            md5 = MessageDigest.getInstance("MD5");
        } catch (NoSuchAlgorithmException e) {
            System.out.println("Error: " + e);
        }
        try {
            long time = System.currentTimeMillis();
            long rand = 0L;
            if (secure) rand = mySecureRand.nextLong(); else rand = myRand.nextLong();
            sbValueBeforeMD5.append(s_id);
            sbValueBeforeMD5.append(":");
            sbValueBeforeMD5.append(Long.toString(time));
            sbValueBeforeMD5.append(":");
            sbValueBeforeMD5.append(Long.toString(rand));
            valueBeforeMD5 = sbValueBeforeMD5.toString();
            md5.update(valueBeforeMD5.getBytes());
            byte array[] = md5.digest();
            StringBuffer sb = new StringBuffer();
            for (int j = 0; j < array.length; j++) {
                int b = array[j] & 0xff;
                if (b < 16) sb.append('0');
                sb.append(Integer.toHexString(b));
            }
            valueAfterMD5 = sb.toString();
        } catch (Exception e) {
            System.out.println("Error:" + e);
        }
    }
```

## P057

**A**

```java
public static String md5String(String string) {
        try {
            MessageDigest msgDigest = MessageDigest.getInstance("MD5");
            msgDigest.update(string.getBytes("UTF-8"));
            byte[] digest = msgDigest.digest();
            String result = "";
            for (int i = 0; i < digest.length; i++) {
                int value = digest[i];
                if (value < 0) value += 256;
                result += Integer.toHexString(value);
            }
            return result;
        } catch (UnsupportedEncodingException error) {
            throw new IllegalArgumentException(error);
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalArgumentException(error);
        }
    }
```

**B**

```java
public static String CreateHash(String s) {
        String str = s.toString();
        if (str == null || str.length() == 0) {
            throw new IllegalArgumentException("String cannot be null or empty");
        }
        StringBuffer hexString = new StringBuffer();
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            md.update(str.getBytes());
            byte[] hash = md.digest();
            for (int i = 0; i < hash.length; i++) {
                if ((0xff & hash[i]) < 0x10) {
                    hexString.append("0" + Integer.toHexString((0xFF & hash[i])));
                } else {
                    hexString.append(Integer.toHexString(0xFF & hash[i]));
                }
            }
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        }
        return (hexString.toString());
    }
```

## P058

**A**

```java
private String generateCode(String seed) {
        try {
            Security.addProvider(new FNVProvider());
            MessageDigest digest = MessageDigest.getInstance("FNV-1a");
            digest.update((seed + UUID.randomUUID().toString()).getBytes());
            byte[] hash1 = digest.digest();
            String sHash1 = "m" + (new String(LibraryBase64.encode(hash1))).replaceAll("=", "").replaceAll("-", "_");
            return sHash1;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return "";
    }
```

**B**

```java
private static String getVisitorId(String guid, String account, String userAgent, Cookie cookie) throws NoSuchAlgorithmException, UnsupportedEncodingException {
        if (cookie != null && cookie.getValue() != null) {
            return cookie.getValue();
        }
        String message;
        if (!isEmpty(guid)) {
            message = guid + account;
        } else {
            message = userAgent + getRandomNumber() + UUID.randomUUID().toString();
        }
        MessageDigest m = MessageDigest.getInstance("MD5");
        m.update(message.getBytes("UTF-8"), 0, message.length());
        byte[] sum = m.digest();
        BigInteger messageAsNumber = new BigInteger(1, sum);
        String md5String = messageAsNumber.toString(16);
        while (md5String.length() < 32) {
            md5String = "0" + md5String;
        }
        return "0x" + md5String.substring(0, 16);
    }
```

## P059

**A**

```java
private void getImage(String filename) throws MalformedURLException, IOException, SAXException, FileNotFoundException {
        String url = Constants.STRATEGICDOMINATION_URL + "/images/gameimages/" + filename;
        WebRequest req = new GetMethodWebRequest(url);
        WebResponse response = wc.getResponse(req);
        File file = new File("etc/images/" + filename);
        FileOutputStream outputStream = new FileOutputStream(file);
        IOUtils.copy(response.getInputStream(), outputStream);
    }
```

**B**

```java
public String getScenarioData(String urlForSalesData) throws IOException, Exception {
        InputStream inputStream = null;
        BufferedReader bufferedReader = null;
        DataInputStream input = null;
        StringBuffer sBuf = new StringBuffer();
        Proxy proxy;
        if (httpWebProxyServer != null && Integer.toString(httpWebProxyPort) != null) {
            SocketAddress address = new InetSocketAddress(httpWebProxyServer, httpWebProxyPort);
            proxy = new Proxy(Proxy.Type.HTTP, address);
        } else {
            proxy = null;
        }
        proxy = null;
        URL url;
        try {
            url = new URL(urlForSalesData);
            HttpURLConnection httpUrlConnection;
            if (proxy != null) httpUrlConnection = (HttpURLConnection) url.openConnection(proxy); else httpUrlConnection = (HttpURLConnection) url.openConnection();
            httpUrlConnection.setDoInput(true);
            httpUrlConnection.setRequestMethod("GET");
            String name = rb.getString("WRAP_NAME");
            String password = rb.getString("WRAP_PASSWORD");
            Credentials simpleAuthCrentials = new Credentials(TOKEN_TYPE.SimpleApiAuthToken, name, password);
            ACSTokenProvider tokenProvider = new ACSTokenProvider(httpWebProxyServer, httpWebProxyPort, simpleAuthCrentials);
            String requestUriStr1 = "https://" + solutionName + "." + acmHostName + "/" + serviceName;
            String appliesTo1 = rb.getString("SIMPLEAPI_APPLIES_TO");
            String token = tokenProvider.getACSToken(requestUriStr1, appliesTo1);
            httpUrlConnection.addRequestProperty("token", "WRAPv0.9 " + token);
            httpUrlConnection.addRequestProperty("solutionName", solutionName);
            httpUrlConnection.connect();
            if (httpUrlConnection.getResponseCode() == HttpServletResponse.SC_UNAUTHORIZED) {
                List<TestSalesOrderService> salesOrderServiceBean = new ArrayList<TestSalesOrderService>();
                TestSalesOrderService response = new TestSalesOrderService();
                response.setResponseCode(HttpServletResponse.SC_UNAUTHORIZED);
                response.setResponseMessage(httpUrlConnection.getResponseMessage());
                salesOrderServiceBean.add(response);
            }
            inputStream = httpUrlConnection.getInputStream();
            input = new DataInputStream(inputStream);
            bufferedReader = new BufferedReader(new InputStreamReader(input));
            String str;
            while (null != ((str = bufferedReader.readLine()))) {
                sBuf.append(str);
            }
            String responseString = sBuf.toString();
            return responseString;
        } catch (MalformedURLException e) {
            throw e;
        } catch (IOException e) {
            throw e;
        }
    }
```

## P060

**A**

```java
private void pack() {
        String szImageDir = m_szBasePath + "Images";
        File fImageDir = new File(szImageDir);
        fImageDir.mkdirs();
        String ljIcon = System.getProperty("user.home");
        ljIcon += System.getProperty("file.separator") + "MochaJournal" + System.getProperty("file.separator") + m_szUsername + System.getProperty("file.separator") + "Cache";
        File fUserDir = new File(ljIcon);
        File[] fIcons = fUserDir.listFiles();
        int iSize = fIcons.length;
        for (int i = 0; i < iSize; i++) {
            try {
                File fOutput = new File(fImageDir, fIcons[i].getName());
                if (!fOutput.exists()) {
                    fOutput.createNewFile();
                    FileOutputStream fOut = new FileOutputStream(fOutput);
                    FileInputStream fIn = new FileInputStream(fIcons[i]);
                    while (fIn.available() > 0) fOut.write(fIn.read());
                }
            } catch (IOException e) {
                System.err.println(e);
            }
        }
        try {
            FileOutputStream fOut;
            InputStream fLJIcon = getClass().getResourceAsStream("/org/homedns/krolain/MochaJournal/Images/userinfo.gif");
            File fLJOut = new File(fImageDir, "user.gif");
            if (!fLJOut.exists()) {
                fOut = new FileOutputStream(fLJOut);
                while (fLJIcon.available() > 0) fOut.write(fLJIcon.read());
            }
            fLJIcon = getClass().getResourceAsStream("/org/homedns/krolain/MochaJournal/Images/communitynfo.gif");
            fLJOut = new File(fImageDir, "comm.gif");
            if (!fLJOut.exists()) {
                fOut = new FileOutputStream(fLJOut);
                while (fLJIcon.available() > 0) fOut.write(fLJIcon.read());
            }
            fLJIcon = getClass().getResourceAsStream("/org/homedns/krolain/MochaJournal/Images/icon_private.gif");
            fLJOut = new File(fImageDir, "icon_private.gif");
            if (!fLJOut.exists()) {
                fOut = new FileOutputStream(fLJOut);
                while (fLJIcon.available() > 0) fOut.write(fLJIcon.read());
            }
            fLJIcon = getClass().getResourceAsStream("/org/homedns/krolain/MochaJournal/Images/icon_protected.gif");
            fLJOut = new File(fImageDir, "icon_protected.gif");
            if (!fLJOut.exists()) {
                fOut = new FileOutputStream(fLJOut);
                while (fLJIcon.available() > 0) fOut.write(fLJIcon.read());
            }
        } catch (IOException e) {
            System.err.println(e);
        }
    }
```

**B**

```java
private void copyFile(String file) {
        FileChannel inChannel = null;
        FileChannel outChannel = null;
        try {
            Date dt = new Date();
            SimpleDateFormat df = new SimpleDateFormat("yyyyMMdd HHmmss ");
            File in = new File(file);
            String[] name = file.split("\\\\");
            File out = new File(".\\xml_archiv\\" + df.format(dt) + name[name.length - 1]);
            inChannel = new FileInputStream(in).getChannel();
            outChannel = new FileOutputStream(out).getChannel();
            inChannel.transferTo(0, inChannel.size(), outChannel);
        } catch (IOException e) {
            System.err.println("Copy error!");
            System.err.println("Error: " + e.getMessage());
        } finally {
            if (inChannel != null) {
                try {
                    inChannel.close();
                } catch (IOException ex) {
                    Logger.getLogger(ImportIntoDb.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
            if (outChannel != null) {
                try {
                    outChannel.close();
                } catch (IOException ex) {
                    Logger.getLogger(ImportIntoDb.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }
    }
```

## P061

**A**

```java
public static void translateTableMetaData(String baseDir, String tableName, NameSpaceDefinition nsDefinition) throws Exception {
        setVosiNS(baseDir, "table", nsDefinition);
        String filename = baseDir + "table.xsl";
        Scanner s = new Scanner(new File(filename));
        PrintWriter fw = new PrintWriter(new File(baseDir + tableName + ".xsl"));
        while (s.hasNextLine()) {
            fw.println(s.nextLine().replaceAll("TABLENAME", tableName));
        }
        s.close();
        fw.close();
        applyStyle(baseDir + "tables.xml", baseDir + tableName + ".json", baseDir + tableName + ".xsl");
    }
```

**B**

```java
private void prepareUrlFile(ZipEntryRef zer, String nodeDir, String reportDir) throws Exception {
        URL url = new URL(zer.getUri());
        URLConnection conn = url.openConnection();
        String fcopyName = reportDir + File.separator + zer.getFilenameFromHttpHeader(conn.getHeaderFields());
        logger.debug("download " + zer.getUri() + " in " + fcopyName);
        BufferedOutputStream bw;
        bw = new BufferedOutputStream(new FileOutputStream(fcopyName));
        BufferedInputStream reader = new BufferedInputStream(conn.getInputStream());
        byte[] inputLine = new byte[100000];
        ;
        while (reader.read(inputLine) > 0) {
            bw.write(inputLine);
        }
        bw.close();
        reader.close();
        zer.setUri(fcopyName);
    }
```

## P062

**A**

```java
public String getCipherString(String source) throws CadenaNoCifradaException {
        String encryptedSource = null;
        MessageDigest md;
        try {
            md = MessageDigest.getInstance("SHA-1");
            byte[] sha1hash = new byte[40];
            md.update(source.getBytes(encoding), 0, source.length());
            sha1hash = md.digest();
            encryptedSource = convertToHex(sha1hash);
        } catch (Exception e) {
            throw new CadenaNoCifradaException(e);
        }
        return encryptedSource;
    }
```

**B**

```java
public static String unsecureHashConstantSalt(String password) throws NoSuchAlgorithmException, UnsupportedEncodingException {
        password = SALT3 + password;
        MessageDigest md5 = MessageDigest.getInstance("MD5");
        md5.update(password.getBytes(), 0, password.length());
        password += convertToHex(md5.digest()) + SALT4;
        MessageDigest md = MessageDigest.getInstance("SHA-512");
        byte[] sha1hash = new byte[40];
        md.update(password.getBytes("UTF-8"), 0, password.length());
        sha1hash = md.digest();
        return convertToHex(sha1hash);
    }
```

## P063

**A**

```java
public boolean copy(File src, File dest, byte[] b) {
        if (src.isDirectory()) {
            String[] ss = src.list();
            for (int i = 0; i < ss.length; i++) if (!copy(new File(src, ss[i]), new File(dest, ss[i]), b)) return false;
            return true;
        }
        delete(dest);
        dest.getParentFile().mkdirs();
        try {
            FileInputStream fis = new FileInputStream(src);
            try {
                FileOutputStream fos = new FileOutputStream(dest);
                try {
                    int read;
                    while ((read = fis.read(b)) != -1) fos.write(b, 0, read);
                } finally {
                    try {
                        fos.close();
                    } catch (IOException ignore) {
                    }
                    register(dest);
                }
            } finally {
                fis.close();
            }
            if (log.isDebugEnabled()) log.debug("Success: M-COPY " + src + " -> " + dest);
            return true;
        } catch (IOException e) {
            log.error("Failed: M-COPY " + src + " -> " + dest, e);
            return false;
        }
    }
```

**B**

```java
@SuppressWarnings("null")
    public static void copyFile(File src, File dst) throws IOException {
        if (!dst.getParentFile().exists()) {
            dst.getParentFile().mkdirs();
        }
        dst.createNewFile();
        FileChannel srcC = null;
        FileChannel dstC = null;
        try {
            srcC = new FileInputStream(src).getChannel();
            dstC = new FileOutputStream(dst).getChannel();
            dstC.transferFrom(srcC, 0, srcC.size());
        } finally {
            try {
                if (dst != null) {
                    dstC.close();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            try {
                if (src != null) {
                    srcC.close();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
```

## P064

**A**

```java
public static void main(String[] args) {
        try {
            FileReader reader = new FileReader(args[0]);
            FileWriter writer = new FileWriter(args[1]);
            html2xhtml(reader, writer);
            writer.close();
            reader.close();
        } catch (Exception e) {
            freemind.main.Resources.getInstance().logException(e);
        }
    }
```

**B**

```java
public static void main(String[] args) {
        RSSReader rssreader = new RSSReader();
        try {
            XmlPullParserFactory factory = XmlPullParserFactory.newInstance();
            XmlPullParser parser = factory.newPullParser();
            String url = args[0];
            InputStreamReader stream = new InputStreamReader(new URL(url).openStream());
            parser.setInput(stream);
            XmlSerializer writer = factory.newSerializer();
            writer.setOutput(new OutputStreamWriter(System.out));
            rssreader.convertRSSToHtml(parser, writer);
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
    }
```

## P065

**A**

```java
private void processar() {
        boolean bOK = false;
        String sSQL = "DELETE FROM FNSALDOLANCA WHERE CODEMP=? AND CODFILIAL=?";
        try {
            state("Excluindo base atual de saldos...");
            PreparedStatement ps = con.prepareStatement(sSQL);
            ps.setInt(1, Aplicativo.iCodEmp);
            ps.setInt(2, ListaCampos.getMasterFilial("FNSALDOLANCA"));
            ps.executeUpdate();
            ps.close();
            state("Base excluida...");
            bOK = true;
        } catch (SQLException err) {
            Funcoes.mensagemErro(this, "Erro ao excluir os saldos!\n" + err.getMessage(), true, con, err);
            err.printStackTrace();
        }
        if (bOK) {
            bOK = false;
            sSQL = "SELECT CODPLAN,DATASUBLANCA,SUM(VLRSUBLANCA) VLRSUBLANCA FROM " + "FNSUBLANCA WHERE CODEMP=? AND CODFILIAL=? GROUP BY CODPLAN,DATASUBLANCA " + "ORDER BY CODPLAN,DATASUBLANCA";
            try {
                state("Iniciando reconstru��o...");
                PreparedStatement ps = con.prepareStatement(sSQL);
                ps.setInt(1, Aplicativo.iCodEmp);
                ps.setInt(2, ListaCampos.getMasterFilial("FNLANCA"));
                ResultSet rs = ps.executeQuery();
                String sPlanAnt = "";
                double dSaldo = 0;
                bOK = true;
                int iFilialPlan = ListaCampos.getMasterFilial("FNPLANEJAMENTO");
                int iFilialSaldo = ListaCampos.getMasterFilial("FNSALDOLANCA");
                while (rs.next() && bOK) {
                    if ("1010100000004".equals(rs.getString("CodPlan"))) {
                        System.out.println("Debug");
                    }
                    if (sPlanAnt.equals(rs.getString("CodPlan"))) {
                        dSaldo += rs.getDouble("VLRSUBLANCA");
                    } else dSaldo = rs.getDouble("VLRSUBLANCA");
                    bOK = insereSaldo(iFilialSaldo, iFilialPlan, rs.getString("CodPlan"), rs.getDate("DataSubLanca"), dSaldo);
                    sPlanAnt = rs.getString("CodPlan");
                    if ("1010100000004".equals(sPlanAnt)) {
                        System.out.println("Debug");
                    }
                }
                ps.close();
                state("Aguardando grava��o final...");
            } catch (SQLException err) {
                bOK = false;
                Funcoes.mensagemErro(this, "Erro ao excluir os lan�amentos!\n" + err.getMessage(), true, con, err);
                err.printStackTrace();
            }
        }
        try {
            if (bOK) {
                con.commit();
                state("Registros processados com sucesso!");
            } else {
                state("Registros antigos restaurados!");
                con.rollback();
            }
        } catch (SQLException err) {
            Funcoes.mensagemErro(this, "Erro ao relizar precedimento!\n" + err.getMessage(), true, con, err);
            err.printStackTrace();
        }
        bRunProcesso = false;
        btProcessar.setEnabled(true);
    }
```

**B**

```java
@Override
    public boolean saveCart(Carrito cart, boolean completado, String date, String formPago) {
        Connection conexion = null;
        PreparedStatement insertHistorial = null;
        PreparedStatement insertCarrito = null;
        boolean exito = false;
        try {
            conexion = pool.getConnection();
            conexion.setAutoCommit(false);
            insertHistorial = conexion.prepareStatement("INSERT INTO " + nameBD + ".HistorialCarritos VALUES (?,?,?,?,?,?)");
            insertHistorial.setString(1, cart.getUser());
            insertHistorial.setString(2, cart.getCodigo());
            insertHistorial.setString(3, date);
            insertHistorial.setDouble(4, cart.getPrecio());
            insertHistorial.setString(5, formPago);
            insertHistorial.setBoolean(6, completado);
            int filasAfectadas = insertHistorial.executeUpdate();
            if (filasAfectadas != 1) {
                conexion.rollback();
            } else {
                insertCarrito = conexion.prepareStatement("INSERT INTO " + nameBD + ".Carritos VALUES (?,?,?,?,?)");
                Iterator<String> iteradorProductos = cart.getArticulos().keySet().iterator();
                while (iteradorProductos.hasNext()) {
                    String key = iteradorProductos.next();
                    Producto prod = getProduct(key);
                    int cantidad = cart.getArticulos().get(key);
                    insertCarrito.setString(1, cart.getCodigo());
                    insertCarrito.setString(2, prod.getCodigo());
                    insertCarrito.setString(3, prod.getNombre());
                    insertCarrito.setDouble(4, prod.getPrecio());
                    insertCarrito.setInt(5, cantidad);
                    filasAfectadas = insertCarrito.executeUpdate();
                    if (filasAfectadas != 1) {
                        conexion.rollback();
                        break;
                    }
                    insertCarrito.clearParameters();
                }
                conexion.commit();
                exito = true;
            }
        } catch (SQLException ex) {
            logger.log(Level.SEVERE, "Error añadiendo carrito al registro", ex);
            try {
                conexion.rollback();
            } catch (SQLException ex1) {
                logger.log(Level.SEVERE, "Error haciendo rollback de la transacción para insertar carrito en el registro", ex1);
            }
        } finally {
            cerrarConexionYStatement(conexion, insertCarrito, insertHistorial);
        }
        return exito;
    }
```

## P066

**A**

```java
private void sendFile(File file, HttpServletResponse response) throws IOException {
        response.setContentLength((int) file.length());
        InputStream inputStream = null;
        try {
            inputStream = new FileInputStream(file);
            IOUtils.copy(inputStream, response.getOutputStream());
        } finally {
            IOUtils.closeQuietly(inputStream);
        }
    }
```

**B**

```java
private void delay(HttpServletRequest request, HttpServletResponse response, FilterChain chain) throws IOException, ServletException {
        String url = request.getRequestURL().toString();
        if (delayed.contains(url)) {
            delayed.remove(url);
            LOGGER.info(MessageFormat.format("Loading delayed resource at url = [{0}]", url));
            chain.doFilter(request, response);
        } else {
            LOGGER.info("Returning resource = [LoaderApplication.swf]");
            InputStream input = null;
            OutputStream output = null;
            try {
                input = getClass().getResourceAsStream("LoaderApplication.swf");
                output = response.getOutputStream();
                delayed.add(url);
                response.setHeader("Cache-Control", "no-cache");
                IOUtils.copy(input, output);
            } finally {
                IOUtils.closeQuietly(output);
                IOUtils.closeQuietly(input);
            }
        }
    }
```

## P067

**A**

```java
public static void copyFile(File src, File dest, boolean preserveFileDate) throws IOException {
        if (src.exists() && src.isDirectory()) {
            throw new IOException("source file exists but is a directory");
        }
        if (dest.exists() && dest.isDirectory()) {
            dest = new File(dest, src.getName());
        }
        if (!dest.exists()) {
            dest.createNewFile();
        }
        FileChannel srcCH = null;
        FileChannel destCH = null;
        try {
            srcCH = new FileInputStream(src).getChannel();
            destCH = new FileOutputStream(dest).getChannel();
            destCH.transferFrom(srcCH, 0, srcCH.size());
        } finally {
            closeQuietly(srcCH);
            closeQuietly(destCH);
        }
        if (src.length() != dest.length()) {
            throw new IOException("Failed to copy full contents from '" + src + "' to '" + dest + "'");
        }
        if (preserveFileDate) {
            dest.setLastModified(src.lastModified());
        }
    }
```

**B**

```java
protected void createFile(File sourceActionDirectory, File destinationActionDirectory, LinkedList<String> segments) throws DuplicateActionFileException {
        File currentSrcDir = sourceActionDirectory;
        File currentDestDir = destinationActionDirectory;
        String segment = "";
        for (int i = 0; i < segments.size() - 1; i++) {
            segment = segments.get(i);
            currentSrcDir = new File(currentSrcDir, segment);
            currentDestDir = new File(currentDestDir, segment);
        }
        if (currentSrcDir != null && currentDestDir != null) {
            File srcFile = new File(currentSrcDir, segments.getLast());
            if (srcFile.exists()) {
                File destFile = new File(currentDestDir, segments.getLast());
                if (destFile.exists()) {
                    throw new DuplicateActionFileException(srcFile.toURI().toASCIIString());
                }
                try {
                    FileChannel srcChannel = new FileInputStream(srcFile).getChannel();
                    FileChannel destChannel = new FileOutputStream(destFile).getChannel();
                    ByteBuffer buffer = ByteBuffer.allocate((int) srcChannel.size());
                    while (srcChannel.position() < srcChannel.size()) {
                        srcChannel.read(buffer);
                    }
                    srcChannel.close();
                    buffer.rewind();
                    destChannel.write(buffer);
                    destChannel.close();
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }
        }
    }
```

## P068

**A**

```java
public static boolean copy(InputStream is, File file) {
        try {
            IOUtils.copy(is, new FileOutputStream(file));
            return true;
        } catch (Exception e) {
            logger.severe(e.getMessage());
            return false;
        }
    }
```

**B**

```java
private void processData(InputStream raw) {
        String fileName = remoteName;
        if (localName != null) {
            fileName = localName;
        }
        try {
            FileOutputStream fos = new FileOutputStream(new File(fileName), true);
            IOUtils.copy(raw, fos);
            LOG.info("ok");
        } catch (IOException e) {
            LOG.error("error writing file", e);
        }
    }
```

## P069

**A**

```java
public boolean loadResource(String resourcePath) {
        try {
            URL url = Thread.currentThread().getContextClassLoader().getResource(resourcePath);
            if (url == null) {
                logger.error("Cannot find the resource named: '" + resourcePath + "'. Failed to load the keyword list.");
                return false;
            }
            InputStreamReader isr = new InputStreamReader(url.openStream());
            BufferedReader br = new BufferedReader(isr);
            String ligne = br.readLine();
            while (ligne != null) {
                if (!contains(ligne.toUpperCase())) addLast(ligne.toUpperCase());
                ligne = br.readLine();
            }
            return true;
        } catch (IOException ioe) {
            logger.log(Level.ERROR, "Cannot load default SQL keywords file.", ioe);
        }
        return false;
    }
```

**B**

```java
private boolean readRemoteFile() {
        InputStream inputstream;
        Concept concept = new Concept();
        try {
            inputstream = url.openStream();
            InputStreamReader inputStreamReader = new InputStreamReader(inputstream);
            BufferedReader bufferedreader = new BufferedReader(inputStreamReader);
            String s4;
            while ((s4 = bufferedreader.readLine()) != null && s4.length() > 0) {
                if (!parseLine(s4, concept)) {
                    return false;
                }
            }
        } catch (MalformedURLException e) {
            logger.fatal("malformed URL, trying to read local file");
            return readLocalFile();
        } catch (IOException e1) {
            logger.fatal("Error reading URL file, trying to read local file");
            return readLocalFile();
        } catch (Exception x) {
            logger.fatal("Failed to readRemoteFile " + x.getMessage() + ", trying to read local file");
            return readLocalFile();
        }
        return true;
    }
```

## P070

**A**

```java
public synchronized void write() throws IOException {
        ZipOutputStream jar = new ZipOutputStream(new FileOutputStream(jarPath));
        int index = className.lastIndexOf('.');
        String packageName = className.substring(0, index);
        String clazz = className.substring(index + 1);
        String directory = packageName.replace('.', '/');
        ZipEntry dummyClass = new ZipEntry(directory + "/" + clazz + ".class");
        jar.putNextEntry(dummyClass);
        ClassGen classgen = new ClassGen(getClassName(), "java.lang.Object", "<generated>", Constants.ACC_PUBLIC | Constants.ACC_SUPER, null);
        byte[] bytes = classgen.getJavaClass().getBytes();
        jar.write(bytes);
        jar.closeEntry();
        ZipEntry synthFile = new ZipEntry(directory + "/synth.xml");
        jar.putNextEntry(synthFile);
        Comment comment = new Comment("Generated by SynthBuilder from L2FProd.com");
        Element root = new Element("synth");
        root.addAttribute(new Attribute("version", "1"));
        root.appendChild(comment);
        Element defaultStyle = new Element("style");
        defaultStyle.addAttribute(new Attribute("id", "default"));
        Element defaultFont = new Element("font");
        defaultFont.addAttribute(new Attribute("name", "SansSerif"));
        defaultFont.addAttribute(new Attribute("size", "12"));
        defaultStyle.appendChild(defaultFont);
        Element defaultState = new Element("state");
        defaultStyle.appendChild(defaultState);
        root.appendChild(defaultStyle);
        Element bind = new Element("bind");
        bind.addAttribute(new Attribute("style", "default"));
        bind.addAttribute(new Attribute("type", "region"));
        bind.addAttribute(new Attribute("key", ".*"));
        root.appendChild(bind);
        doc = new Document(root);
        imagesToCopy = new HashMap();
        ComponentStyle[] styles = config.getStyles();
        for (ComponentStyle element : styles) {
            write(element);
        }
        Serializer writer = new Serializer(jar);
        writer.setIndent(2);
        writer.write(doc);
        writer.flush();
        jar.closeEntry();
        for (Iterator iter = imagesToCopy.keySet().iterator(); iter.hasNext(); ) {
            String element = (String) iter.next();
            File pathToImage = (File) imagesToCopy.get(element);
            ZipEntry image = new ZipEntry(directory + "/" + element);
            jar.putNextEntry(image);
            FileInputStream input = new FileInputStream(pathToImage);
            int read = -1;
            while ((read = input.read()) != -1) {
                jar.write(read);
            }
            input.close();
            jar.flush();
            jar.closeEntry();
        }
        jar.flush();
        jar.close();
    }
```

**B**

```java
public int process(ProcessorContext context) throws InterruptedException, ProcessorException {
        logger.info("JAISaveTask:process");
        final RenderedOp im = (RenderedOp) context.get("RenderedOp");
        final String path = "s3://s3.amazonaws.com/rssfetch/" + (new Guid());
        final PNGEncodeParam.RGB encPar = new PNGEncodeParam.RGB();
        encPar.setTransparentRGB(new int[] { 0, 0, 0 });
        File tmpFile = null;
        try {
            tmpFile = File.createTempFile("thmb", ".png");
            OutputStream out = new FileOutputStream(tmpFile);
            final ParameterBlock pb = (new ParameterBlock()).addSource(im).add(out).add("png").add(encPar);
            JAI.create("encode", pb, null);
            out.flush();
            out.close();
            FileInputStream in = new FileInputStream(tmpFile);
            final XFile xfile = new XFile(path);
            final XFileOutputStream xout = new XFileOutputStream(xfile);
            final com.luzan.common.nfs.s3.XFileExtensionAccessor xfa = ((com.luzan.common.nfs.s3.XFileExtensionAccessor) xfile.getExtensionAccessor());
            if (xfa != null) {
                xfa.setMimeType("image/png");
                xfa.setContentLength(tmpFile.length());
            }
            IOUtils.copy(in, xout);
            xout.flush();
            xout.close();
            in.close();
            context.put("outputPath", path);
        } catch (IOException e) {
            logger.error(e);
            throw new ProcessorException(e);
        } catch (Throwable e) {
            logger.error(e);
            throw new ProcessorException(e);
        } finally {
            if (tmpFile != null && tmpFile.exists()) {
                tmpFile.delete();
            }
        }
        return TaskState.STATE_MO_START + TaskState.STATE_ENCODE;
    }
```

## P071

**A**

```java
public void writeValue(Value v) throws IOException, SQLException {
        int type = v.getType();
        writeInt(type);
        switch(type) {
            case Value.NULL:
                break;
            case Value.BYTES:
            case Value.JAVA_OBJECT:
                writeBytes(v.getBytesNoCopy());
                break;
            case Value.UUID:
                {
                    ValueUuid uuid = (ValueUuid) v;
                    writeLong(uuid.getHigh());
                    writeLong(uuid.getLow());
                    break;
                }
            case Value.BOOLEAN:
                writeBoolean(v.getBoolean().booleanValue());
                break;
            case Value.BYTE:
                writeByte(v.getByte());
                break;
            case Value.TIME:
                writeLong(v.getTimeNoCopy().getTime());
                break;
            case Value.DATE:
                writeLong(v.getDateNoCopy().getTime());
                break;
            case Value.TIMESTAMP:
                {
                    Timestamp ts = v.getTimestampNoCopy();
                    writeLong(ts.getTime());
                    writeInt(ts.getNanos());
                    break;
                }
            case Value.DECIMAL:
                writeString(v.getString());
                break;
            case Value.DOUBLE:
                writeDouble(v.getDouble());
                break;
            case Value.FLOAT:
                writeFloat(v.getFloat());
                break;
            case Value.INT:
                writeInt(v.getInt());
                break;
            case Value.LONG:
                writeLong(v.getLong());
                break;
            case Value.SHORT:
                writeInt(v.getShort());
                break;
            case Value.STRING:
            case Value.STRING_IGNORECASE:
            case Value.STRING_FIXED:
                writeString(v.getString());
                break;
            case Value.BLOB:
                {
                    long length = v.getPrecision();
                    if (SysProperties.CHECK && length < 0) {
                        Message.throwInternalError("length: " + length);
                    }
                    writeLong(length);
                    InputStream in = v.getInputStream();
                    long written = IOUtils.copyAndCloseInput(in, out);
                    if (SysProperties.CHECK && written != length) {
                        Message.throwInternalError("length:" + length + " written:" + written);
                    }
                    writeInt(LOB_MAGIC);
                    break;
                }
            case Value.CLOB:
                {
                    long length = v.getPrecision();
                    if (SysProperties.CHECK && length < 0) {
                        Message.throwInternalError("length: " + length);
                    }
                    writeLong(length);
                    Reader reader = v.getReader();
                    java.io.OutputStream out2 = new java.io.FilterOutputStream(out) {

                        public void flush() {
                        }
                    };
                    Writer writer = new BufferedWriter(new OutputStreamWriter(out2, Constants.UTF8));
                    long written = IOUtils.copyAndCloseInput(reader, writer);
                    if (SysProperties.CHECK && written != length) {
                        Message.throwInternalError("length:" + length + " written:" + written);
                    }
                    writer.flush();
                    writeInt(LOB_MAGIC);
                    break;
                }
            case Value.ARRAY:
                {
                    Value[] list = ((ValueArray) v).getList();
                    writeInt(list.length);
                    for (Value value : list) {
                        writeValue(value);
                    }
                    break;
                }
            case Value.RESULT_SET:
                {
                    ResultSet rs = ((ValueResultSet) v).getResultSet();
                    rs.beforeFirst();
                    ResultSetMetaData meta = rs.getMetaData();
                    int columnCount = meta.getColumnCount();
                    writeInt(columnCount);
                    for (int i = 0; i < columnCount; i++) {
                        writeString(meta.getColumnName(i + 1));
                        writeInt(meta.getColumnType(i + 1));
                        writeInt(meta.getPrecision(i + 1));
                        writeInt(meta.getScale(i + 1));
                    }
                    while (rs.next()) {
                        writeBoolean(true);
                        for (int i = 0; i < columnCount; i++) {
                            int t = DataType.convertSQLTypeToValueType(meta.getColumnType(i + 1));
                            Value val = DataType.readValue(session, rs, i + 1, t);
                            writeValue(val);
                        }
                    }
                    writeBoolean(false);
                    rs.beforeFirst();
                    break;
                }
            default:
                Message.throwInternalError("type=" + type);
        }
    }
```

**B**

```java
protected String loadPage(String url_string) {
        try {
            URL url = new URL(url_string);
            HttpURLConnection connection = null;
            InputStream is = null;
            try {
                connection = (HttpURLConnection) url.openConnection();
                int response = connection.getResponseCode();
                if (response == HttpURLConnection.HTTP_ACCEPTED || response == HttpURLConnection.HTTP_OK) {
                    is = connection.getInputStream();
                    String page = "";
                    while (page.length() < MAX_PAGE_SIZE) {
                        byte[] buffer = new byte[2048];
                        int len = is.read(buffer);
                        if (len < 0) {
                            break;
                        }
                        page += new String(buffer, 0, len);
                    }
                    return (page);
                } else {
                    informFailure("httpinvalidresponse", "" + response);
                    return (null);
                }
            } finally {
                try {
                    if (is != null) {
                        is.close();
                    }
                    if (connection != null) {
                        connection.disconnect();
                    }
                } catch (Throwable e) {
                    Debug.printStackTrace(e);
                }
            }
        } catch (Throwable e) {
            informFailure("httploadfail", e.toString());
            return (null);
        }
    }
```

## P072

**A**

```java
public void elimina(Pedido pe) throws errorSQL, errorConexionBD {
        System.out.println("GestorPedido.elimina()");
        int id = pe.getId();
        String sql;
        Statement stmt = null;
        try {
            gd.begin();
            sql = "DELETE FROM pedido WHERE id=" + id;
            System.out.println("Ejecutando: " + sql);
            stmt = gd.getConexion().createStatement();
            stmt.executeUpdate(sql);
            System.out.println("executeUpdate");
            gd.commit();
            System.out.println("commit");
            stmt.close();
        } catch (SQLException e) {
            gd.rollback();
            throw new errorSQL(e.toString());
        } catch (errorConexionBD e) {
            System.err.println("Error en GestorPedido.elimina(): " + e);
        } catch (errorSQL e) {
            System.err.println("Error en GestorPedido.elimina(): " + e);
        }
    }
```

**B**

```java
public boolean crear() {
        int result = 0;
        String sql = "insert into divisionxTorneo" + "(torneo_idTorneo, tipoTorneo_idTipoTorneo, nombreDivision, descripcion, numJugadores, numFechas, terminado, tipoDesempate, rondaActual, ptosxbye)" + "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
        try {
            connection = conexionBD.getConnection();
            connection.setAutoCommit(false);
            ps = connection.prepareStatement(sql);
            populatePreparedStatement();
            result = ps.executeUpdate();
            connection.commit();
        } catch (SQLException ex) {
            ex.printStackTrace();
            try {
                connection.rollback();
            } catch (SQLException exe) {
                exe.printStackTrace();
            }
        } finally {
            conexionBD.close(ps);
            conexionBD.close(connection);
        }
        return (result > 0);
    }
```

## P073

**A**

```java
public void xtest1() throws Exception {
        InputStream input = new FileInputStream("C:/Documentos/j931_01.pdf");
        InputStream tmp = new ITextManager().cut(input, 3, 8);
        FileOutputStream output = new FileOutputStream("C:/temp/split.pdf");
        IOUtils.copy(tmp, output);
        input.close();
        tmp.close();
        output.close();
    }
```

**B**

```java
public final void testT4CClientWriter() throws Exception {
        InputStream is = ClassLoader.getSystemResourceAsStream(this.testFileName);
        T4CClientReader reader = new T4CClientReader(is, rc);
        File tmpFile = File.createTempFile("barde", ".log", this.tmpDir);
        System.out.println("tmp=" + tmpFile.getAbsolutePath());
        T4CClientWriter writer = new T4CClientWriter(new FileOutputStream(tmpFile), rc);
        for (Message m = reader.read(); m != null; m = reader.read()) writer.write(m);
        writer.close();
        InputStream fa = ClassLoader.getSystemResourceAsStream(this.testFileName);
        FileInputStream fb = new FileInputStream(tmpFile);
		for (int ba = fa.read(); ba != -1; ba = fa.read()) assertEquals(ba, fb.read());
    }
```

## P074

**A**

```java
public static String hash(final String s) {
        if (s == null || s.length() == 0) return null;
        try {
            final MessageDigest hashEngine = MessageDigest.getInstance("SHA-1");
            hashEngine.update(s.getBytes("iso-8859-1"), 0, s.length());
            return convertToHex(hashEngine.digest());
        } catch (final Exception e) {
            return null;
        }
    }
```

**B**

```java
public static String encryptPasswd(String pass) {
        try {
            if (pass == null || pass.length() == 0) return pass;
            MessageDigest sha = MessageDigest.getInstance("SHA-1");
            sha.reset();
            sha.update(pass.getBytes("UTF-8"));
            return Base64OutputStream.encode(sha.digest());
        } catch (Throwable t) {
            throw new SystemException(t);
        }
    }
```

## P075

**A**

```java
private static String md5(String pwd) {
        try {
            MessageDigest md5 = MessageDigest.getInstance("MD5");
            md5.update(pwd.getBytes(), 0, pwd.length());
            return new BigInteger(1, md5.digest()).toString(16);
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
            throw new Error();
        }
    }
```

**B**

```java
public static String createHash(String password) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            md.update(password.getBytes());
            byte[] digest = md.digest();
            return toHexString(digest);
        } catch (NoSuchAlgorithmException nsae) {
            System.out.println(nsae.getMessage());
        }
        return "";
    }
```

## P076

**A**

```java
public Object execute(ExecutionEvent event) throws ExecutionException {
        final List<InformationUnit> informationUnitsFromExecutionEvent = InformationHandlerUtil.getInformationUnitsFromExecutionEvent(event);
        Shell activeShell = HandlerUtil.getActiveShell(event);
        DirectoryDialog fd = new DirectoryDialog(activeShell, SWT.SAVE);
        String section = Activator.getDefault().getDialogSettings().get("lastExportSection");
        fd.setFilterPath(section);
        final String open = fd.open();
        if (open != null) {
            Activator.getDefault().getDialogSettings().put("lastExportSection", open);
            CancelableRunnable runnable = new CancelableRunnable() {

                @Override
                protected IStatus runCancelableRunnable(IProgressMonitor monitor) {
                    IStatus returnValue = Status.OK_STATUS;
                    monitor.beginTask(NLS.bind(Messages.SaveFileOnDiskHandler_SavingFiles, open), informationUnitsFromExecutionEvent.size());
                    for (InformationUnit informationUnit : informationUnitsFromExecutionEvent) {
                        if (!monitor.isCanceled()) {
                            monitor.setTaskName(NLS.bind(Messages.SaveFileOnDiskHandler_Saving, informationUnit.getLabel()));
                            InformationStructureRead read = InformationStructureRead.newSession(informationUnit);
                            read.getValueByNodeId(Activator.FILENAME);
                            IFile binaryReferenceFile = InformationUtil.getBinaryReferenceFile(informationUnit);
                            FileWriter writer = null;
                            try {
                                if (binaryReferenceFile != null) {
                                    File file = new File(open, (String) read.getValueByNodeId(Activator.FILENAME));
                                    InputStream contents = binaryReferenceFile.getContents();
                                    writer = new FileWriter(file);
                                    IOUtils.copy(contents, writer);
                                    monitor.worked(1);
                                }
                            } catch (Exception e) {
                                returnValue = StatusCreator.newStatus(NLS.bind(Messages.SaveFileOnDiskHandler_ErrorSaving, informationUnit.getLabel(), e));
                                break;
                            } finally {
                                if (writer != null) {
                                    try {
                                        writer.flush();
                                        writer.close();
                                    } catch (IOException e) {
                                    }
                                }
                            }
                        }
                    }
                    return returnValue;
                }
            };
            ProgressMonitorDialog progressMonitorDialog = new ProgressMonitorDialog(activeShell);
            try {
                progressMonitorDialog.run(true, true, runnable);
            } catch (InvocationTargetException e) {
                if (e.getCause() instanceof CoreException) {
                    ErrorDialog.openError(activeShell, Messages.SaveFileOnDiskHandler_ErrorSaving2, Messages.SaveFileOnDiskHandler_ErrorSaving2, ((CoreException) e.getCause()).getStatus());
                } else {
                    ErrorDialog.openError(activeShell, Messages.SaveFileOnDiskHandler_ErrorSaving2, Messages.SaveFileOnDiskHandler_ErrorSaving2, StatusCreator.newStatus(Messages.SaveFileOnDiskHandler_ErrorSaving3, e));
                }
            } catch (InterruptedException e) {
            }
        }
        return null;
    }
```

**B**

```java
@Override
    protected IProject createProject(String projectName, IProgressMonitor monitor) throws CoreException {
        monitor.beginTask(CheatSheetsPlugin.INSTANCE.getString("_UI_CreateJavaProject_message", new String[] { projectName }), 5);
        IProject project = super.createProject(projectName, new SubProgressMonitor(monitor, 1));
        if (project != null) {
            IProjectDescription description = project.getDescription();
            if (!description.hasNature(JavaCore.NATURE_ID)) {
                IJavaProject javaProject = JavaCore.create(project);
                if (javaProject != null) {
                    String[] natures = description.getNatureIds();
                    String[] javaNatures = new String[natures.length + 1];
                    System.arraycopy(natures, 0, javaNatures, 0, natures.length);
                    javaNatures[natures.length] = JavaCore.NATURE_ID;
                    description.setNatureIds(javaNatures);
                    project.setDescription(description, new SubProgressMonitor(monitor, 1));
                    IFolder sourceFolder = project.getFolder(SOURCE_FOLDER);
                    if (!sourceFolder.exists()) {
                        sourceFolder.create(true, true, new SubProgressMonitor(monitor, 1));
                    }
                    javaProject.setOutputLocation(project.getFolder(OUTPUT_FOLDER).getFullPath(), new SubProgressMonitor(monitor, 1));
                    IClasspathEntry[] entries = new IClasspathEntry[] { JavaCore.newSourceEntry(sourceFolder.getFullPath()), JavaCore.newContainerEntry(new Path("org.eclipse.jdt.launching.JRE_CONTAINER")) };
                    javaProject.setRawClasspath(entries, new SubProgressMonitor(monitor, 1));
                }
            }
        }
        monitor.done();
        return project;
    }
```

## P077

**A**

```java
private String getTextResponse(String address) throws Exception {
        URL url = new URL(address);
        HttpURLConnection con = (HttpURLConnection) url.openConnection();
        con.setUseCaches(false);
        BufferedReader in = null;
        try {
            con.connect();
            assertEquals(HttpURLConnection.HTTP_OK, con.getResponseCode());
            in = new BufferedReader(new InputStreamReader(con.getInputStream()));
            StringBuilder builder = new StringBuilder();
            String inputLine = null;
            while ((inputLine = in.readLine()) != null) {
                builder.append(inputLine);
            }
            return builder.toString();
        } finally {
            if (in != null) {
                in.close();
            }
            con.disconnect();
        }
    }
```

**B**

```java
public String sendXml(URL url, String xmlMessage, boolean isResponseExpected) throws IOException {
        if (url == null) {
            throw new IllegalArgumentException("url == null");
        }
        if (xmlMessage == null) {
            throw new IllegalArgumentException("xmlMessage == null");
        }
        LOGGER.finer("url = " + url);
        LOGGER.finer("xmlMessage = :" + xmlMessage + ":");
        LOGGER.finer("isResponseExpected = " + isResponseExpected);
        String answer = null;
        try {
            URLConnection urlConnection = url.openConnection();
            urlConnection.setRequestProperty("Content-type", "text/xml");
            urlConnection.setDoOutput(true);
            urlConnection.setUseCaches(false);
            Writer writer = null;
            try {
                writer = new OutputStreamWriter(urlConnection.getOutputStream());
                writer.write(xmlMessage);
                writer.flush();
            } finally {
                if (writer != null) {
                    writer.close();
                }
            }
            LOGGER.finer("message written");
            StringBuilder sb = new StringBuilder();
            BufferedReader in = null;
            try {
                in = new BufferedReader(new InputStreamReader(urlConnection.getInputStream()));
                if (isResponseExpected) {
                    String inputLine;
                    while ((inputLine = in.readLine()) != null) {
                        sb.append(inputLine).append("\n");
                    }
                    answer = sb.toString();
                    LOGGER.finer("response read");
                }
            } catch (FileNotFoundException e) {
                LOGGER.log(Level.SEVERE, "No response", e);
            } finally {
                if (in != null) {
                    in.close();
                }
            }
        } catch (ConnectException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
        }
        LOGGER.finer("answer = :" + answer + ":");
        return answer;
    }
```

## P078

**A**

```java
public static void main(String[] args) throws NoSuchAlgorithmException {
        String password = "root";
        MessageDigest messageDigest = MessageDigest.getInstance("MD5");
        messageDigest.update(password.getBytes());
        final byte[] digest = messageDigest.digest();
        final StringBuilder buf = new StringBuilder(digest.length * 2);
        for (int j = 0; j < digest.length; j++) {
            buf.append(HEX_DIGITS[(digest[j] >> 4) & 0x0f]);
            buf.append(HEX_DIGITS[digest[j] & 0x0f]);
        }
        String pwd = buf.toString();
        System.out.println(pwd);
    }
```

**B**

```java
private void getRandomGUID(boolean secure) throws NoSuchAlgorithmException {
        MessageDigest md5 = null;
        StringBuffer sbValueBeforeMD5 = new StringBuffer();
        try {
            md5 = MessageDigest.getInstance("MD5");
        } catch (NoSuchAlgorithmException e) {
            System.out.println("Error: " + e);
            throw e;
        }
        try {
            long time = System.currentTimeMillis();
            long rand = 0;
            if (secure) {
                rand = mySecureRand.nextLong();
            } else {
                rand = myRand.nextLong();
            }
            sbValueBeforeMD5.append(s_id);
            sbValueBeforeMD5.append(":");
            sbValueBeforeMD5.append(Long.toString(time));
            sbValueBeforeMD5.append(":");
            sbValueBeforeMD5.append(Long.toString(rand));
            valueBeforeMD5 = sbValueBeforeMD5.toString();
            md5.update(valueBeforeMD5.getBytes());
            byte[] array = md5.digest();
            StringBuffer sb = new StringBuffer();
            for (int j = 0; j < array.length; ++j) {
                int b = array[j] & 0xFF;
                if (b < 0x10) sb.append('0');
                sb.append(Integer.toHexString(b));
            }
            valueAfterMD5 = sb.toString();
        } catch (Exception e) {
            System.out.println("Error:" + e);
        }
    }
```

## P079

**A**

```java
private static void main(String[] args) {
        try {
            File f = new File("test.txt");
            if (f.exists()) {
                throw new IOException(f + " already exists.  I don't want to overwrite it.");
            }
            StraightStreamReader in;
            char[] cbuf = new char[0x1000];
            int read;
            int totRead;
            FileOutputStream out = new FileOutputStream(f);
            for (int i = 0x00; i < 0x100; i++) {
                out.write(i);
            }
            out.close();
            in = new StraightStreamReader(new FileInputStream(f));
            for (int i = 0x00; i < 0x100; i++) {
                read = in.read();
                if (read != i) {
                    System.err.println("Error: " + i + " read as " + read);
                }
            }
            in.close();
            in = new StraightStreamReader(new FileInputStream(f));
            totRead = in.read(cbuf);
            if (totRead != 0x100) {
                System.err.println("Simple buffered read did not read the full amount: 0x" + Integer.toHexString(totRead));
            }
            for (int i = 0x00; i < totRead; i++) {
                if (cbuf[i] != i) {
                    System.err.println("Error: 0x" + i + " read as 0x" + cbuf[i]);
                }
            }
            in.close();
            in = new StraightStreamReader(new FileInputStream(f));
            totRead = 0;
            while (totRead <= 0x100 && (read = in.read(cbuf, totRead, 0x100 - totRead)) > 0) {
                totRead += read;
            }
            if (totRead != 0x100) {
                System.err.println("Not enough read. Bytes read: " + Integer.toHexString(totRead));
            }
            for (int i = 0x00; i < totRead; i++) {
                if (cbuf[i] != i) {
                    System.err.println("Error: 0x" + i + " read as 0x" + cbuf[i]);
                }
            }
            in.close();
            in = new StraightStreamReader(new FileInputStream(f));
            totRead = 0;
            while (totRead <= 0x100 && (read = in.read(cbuf, totRead + 0x123, 0x100 - totRead)) > 0) {
                totRead += read;
            }
            if (totRead != 0x100) {
                System.err.println("Not enough read. Bytes read: " + Integer.toHexString(totRead));
            }
            for (int i = 0x00; i < totRead; i++) {
                if (cbuf[i + 0x123] != i) {
                    System.err.println("Error: 0x" + i + " read as 0x" + cbuf[i + 0x123]);
                }
            }
            in.close();
            in = new StraightStreamReader(new FileInputStream(f));
            totRead = 0;
            while (totRead <= 0x100 && (read = in.read(cbuf, totRead + 0x123, 7)) > 0) {
                totRead += read;
            }
            if (totRead != 0x100) {
                System.err.println("Not enough read. Bytes read: " + Integer.toHexString(totRead));
            }
            for (int i = 0x00; i < totRead; i++) {
                if (cbuf[i + 0x123] != i) {
                    System.err.println("Error: 0x" + i + " read as 0x" + cbuf[i + 0x123]);
                }
            }
            in.close();
            f.delete();
        } catch (IOException x) {
            System.err.println(x.getMessage());
        }
    }
```

**B**

```java
public static String postServiceContent(String serviceURL, String text) throws IOException {
        URL url = new URL(serviceURL);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("POST");
        connection.connect();
        int code = connection.getResponseCode();
        if (code == HttpURLConnection.HTTP_OK) {
            InputStream is = connection.getInputStream();
            byte[] buffer = null;
            String stringBuffer = "";
            buffer = new byte[4096];
            int totBytes, bytes, sumBytes = 0;
            totBytes = connection.getContentLength();
            while (true) {
                bytes = is.read(buffer);
                if (bytes <= 0) break;
                stringBuffer = stringBuffer + new String(buffer);
            }
            return stringBuffer;
        }
        return null;
    }
```

## P080

**A**

```java
private void getRandomGUID(boolean secure) {
        MessageDigest md5 = null;
        StringBuffer sbValueBeforeMD5 = new StringBuffer();
        try {
            md5 = MessageDigest.getInstance("MD5");
        } catch (NoSuchAlgorithmException e) {
            logger.debug("Random GUID error: " + e.getMessage());
        }
        try {
            long time = System.currentTimeMillis();
            long rand = 0;
            if (secure) {
                rand = mySecureRand.nextLong();
            } else {
                rand = myRand.nextLong();
            }
            sbValueBeforeMD5.append(s_id);
            sbValueBeforeMD5.append(":");
            sbValueBeforeMD5.append(Long.toString(time));
            sbValueBeforeMD5.append(":");
            sbValueBeforeMD5.append(Long.toString(rand));
            valueBeforeMD5 = sbValueBeforeMD5.toString();
            md5.update(valueBeforeMD5.getBytes());
            byte[] array = md5.digest();
            StringBuffer sb = new StringBuffer();
            for (int j = 0; j < array.length; ++j) {
                int b = array[j] & 0xFF;
                if (b < 0x10) sb.append('0');
                sb.append(Integer.toHexString(b));
            }
            valueAfterMD5 = sb.toString();
        } catch (Exception e) {
            System.out.println("Error:" + e);
        }
    }
```

**B**

```java
private void generate(String salt) {
        MessageDigest md5 = null;
        StringBuffer sbValueBeforeMD5 = new StringBuffer();
        try {
            md5 = MessageDigest.getInstance("MD5");
        } catch (NoSuchAlgorithmException e) {
            logger.error("No MD5", e);
        }
        long time = System.currentTimeMillis();
        long rand = random.nextLong();
        sbValueBeforeMD5.append(systemId);
        sbValueBeforeMD5.append(salt);
        sbValueBeforeMD5.append(Long.toString(time));
        sbValueBeforeMD5.append(Long.toString(rand));
        md5.update(sbValueBeforeMD5.toString().getBytes());
        byte[] array = md5.digest();
        StringBuffer sb = new StringBuffer();
        int position = 0;
        for (int j = 0; j < array.length; ++j) {
            if (position == 4 || position == 6 || position == 8 || position == 10) {
                sb.append('-');
            }
            position++;
            int b = array[j] & 0xFF;
            if (b < 0x10) {
                sb.append('0');
            }
            sb.append(Integer.toHexString(b).toUpperCase());
        }
        guidString = sb.toString().toUpperCase();
    }
```

## P081

**A**

```java
@Test
    public void testWrite() throws Exception {
        MrstkXmlFileReader reader = new MrstkXmlFileReader();
        reader.setFileName("..//data//MrstkXML//prototype3.xml");
        reader.read();
        SpectrumArray sp = reader.getOutput();
        File tmp = File.createTempFile("mrstktest", ".xml");
        System.out.println("Writing temp file: " + tmp.getAbsolutePath());
        MrstkXmlFileWriter writer = new MrstkXmlFileWriter(sp);
        writer.setFile(tmp);
        writer.write();
        MrstkXmlFileReader reader2 = new MrstkXmlFileReader();
        reader2.setFileName(writer.getFile().getAbsolutePath());
        reader2.read();
        SpectrumArray sp2 = reader2.getOutput();
        assertTrue(sp.equals(sp2));
    }
```

**B**

```java
public void doPost(HttpServletRequest req, HttpServletResponse res) throws ServletException, IOException {
        PrintWriter out = null;
        ServletOutputStream outstream = null;
        try {
            String action = req.getParameter("nmrshiftdbaction");
            String relativepath = ServletUtils.expandRelative(this.getServletConfig(), "/WEB-INF");
            TurbineConfig tc = new TurbineConfig(relativepath + "..", relativepath + getServletConfig().getInitParameter("properties"));
            tc.init();
            int spectrumId = -1;
            DBSpectrum spectrum = null;
            Export export = null;
            String format = req.getParameter("format");
            if (action.equals("test")) {
                try {
                    res.setContentType("text/plain");
                    out = res.getWriter();
                    List l = DBSpectrumPeer.executeQuery("select SPECTRUM_ID from SPECTRUM limit 1");
                    if (l.size() > 0) spectrumId = ((Record) l.get(0)).getValue(1).asInt();
                    out.write("success");
                } catch (Exception ex) {
                    out.write("failure");
                }
            } else if (action.equals("rss")) {
                int numbertoexport = 10;
                out = res.getWriter();
                if (req.getParameter("numbertoexport") != null) {
                    try {
                        numbertoexport = Integer.parseInt(req.getParameter("numbertoexport"));
                        if (numbertoexport < 1 || numbertoexport > 20) throw new NumberFormatException("Number to small/large");
                    } catch (NumberFormatException ex) {
                        out.println("The parameter <code>numbertoexport</code>must be an integer from 1 to 20");
                    }
                }
                res.setContentType("text/xml");
                RssWriter rssWriter = new RssWriter();
                rssWriter.setWriter(res.getWriter());
                AtomContainerSet soac = new AtomContainerSet();
                String query = "select distinct MOLECULE.MOLECULE_ID from MOLECULE, SPECTRUM where SPECTRUM.MOLECULE_ID = MOLECULE.MOLECULE_ID and SPECTRUM.REVIEW_FLAG =\"true\" order by MOLECULE.DATE desc;";
                List l = NmrshiftdbUserPeer.executeQuery(query);
                for (int i = 0; i < numbertoexport; i++) {
                    if (i == l.size()) break;
                    DBMolecule mol = DBMoleculePeer.retrieveByPK(new NumberKey(((Record) l.get(i)).getValue(1).asInt()));
                    IMolecule cdkmol = mol.getAsCDKMoleculeAsEntered(1);
                    soac.addAtomContainer(cdkmol);
                    rssWriter.getLinkmap().put(cdkmol, mol.getEasylink(req));
                    rssWriter.getDatemap().put(cdkmol, mol.getDate());
                    rssWriter.getTitlemap().put(cdkmol, mol.getChemicalNamesAsOneStringWithFallback());
                    rssWriter.getCreatormap().put(cdkmol, mol.getNmrshiftdbUser().getUserName());
                    rssWriter.setCreator(GeneralUtils.getAdminEmail(getServletConfig()));
                    Vector v = mol.getDBCanonicalNames();
                    for (int k = 0; k < v.size(); k++) {
                        DBCanonicalName canonName = (DBCanonicalName) v.get(k);
                        if (canonName.getDBCanonicalNameType().getCanonicalNameType() == "INChI") {
                            rssWriter.getInchimap().put(cdkmol, canonName.getName());
                            break;
                        }
                    }
                    rssWriter.setTitle("NMRShiftDB");
                    rssWriter.setLink("http://www.nmrshiftdb.org");
                    rssWriter.setDescription("NMRShiftDB is an open-source, open-access, open-submission, open-content web database for chemical structures and their nuclear magnetic resonance data");
                    rssWriter.setPublisher("NMRShiftDB.org");
                    rssWriter.setImagelink("http://www.nmrshiftdb.org/images/nmrshift-logo.gif");
                    rssWriter.setAbout("http://www.nmrshiftdb.org/NmrshiftdbServlet?nmrshiftdbaction=rss");
                    Collection coll = new ArrayList();
                    Vector spectra = mol.selectSpectra(null);
                    for (int k = 0; k < spectra.size(); k++) {
                        Element el = ((DBSpectrum) spectra.get(k)).getCmlSpect();
                        Element el2 = el.getChildElements().get(0);
                        el.removeChild(el2);
                        coll.add(el2);
                    }
                    rssWriter.getMultiMap().put(cdkmol, coll);
                }
                rssWriter.write(soac);
            } else if (action.equals("getattachment")) {
                res.setContentType("application/zip");
                outstream = res.getOutputStream();
                DBSample sample = DBSamplePeer.retrieveByPK(new NumberKey(req.getParameter("sampleid")));
                outstream.write(sample.getAttachment());
            } else if (action.equals("createreport")) {
                res.setContentType("application/pdf");
                outstream = res.getOutputStream();
                boolean yearly = req.getParameter("style").equals("yearly");
                int yearstart = Integer.parseInt(req.getParameter("yearstart"));
                int yearend = Integer.parseInt(req.getParameter("yearend"));
                int monthstart = 0;
                int monthend = 0;
                if (!yearly) {
                    monthstart = Integer.parseInt(req.getParameter("monthstart"));
                    monthend = Integer.parseInt(req.getParameter("monthend"));
                }
                int type = Integer.parseInt(req.getParameter("type"));
                JasperReport jasperReport = (JasperReport) JRLoader.loadObject(relativepath + "/reports/" + (yearly ? "yearly" : "monthly") + "_report_" + type + ".jasper");
                Map parameters = new HashMap();
                if (yearly) parameters.put("HEADER", "Report for years " + yearstart + " - " + yearend); else parameters.put("HEADER", "Report for " + monthstart + "/" + yearstart + " - " + monthend + "/" + yearend);
                DBConnection dbconn = TurbineDB.getConnection();
                Connection conn = dbconn.getConnection();
                Statement stmt = conn.createStatement();
                ResultSet rs = null;
                if (type == 1) {
                    rs = stmt.executeQuery("select YEAR(DATE) as YEAR, " + (yearly ? "" : " MONTH(DATE) as MONTH, ") + "AFFILIATION_1, AFFILIATION_2, MACHINE.NAME as NAME, count(*) as C, sum(WISHED_SPECTRUM like '%13C%' or WISHED_SPECTRUM like '%variable temperature%' or WISHED_SPECTRUM like '%ID sel. NOE%' or WISHED_SPECTRUM like '%solvent suppression%' or WISHED_SPECTRUM like '%standard spectrum%') as 1_D, sum(WISHED_SPECTRUM like '%H,H-COSY%' or WISHED_SPECTRUM like '%NOESY%' or WISHED_SPECTRUM like '%HMQC%' or WISHED_SPECTRUM like '%HMBC%') as 2_D, sum(OTHER_WISHED_SPECTRUM!='') as SPECIAL, sum(OTHER_NUCLEI!='') as HETERO, sum(PROCESS='self') as SELF, sum(PROCESS='robot') as ROBOT, sum(PROCESS='worker') as OPERATOR from (SAMPLE join TURBINE_USER using (USER_ID)) join MACHINE on MACHINE.MACHINE_ID=SAMPLE.MACHINE where YEAR(DATE)>=" + yearstart + " and YEAR(DATE)<=" + yearend + " and LOGIN_NAME<>'testuser' group by YEAR, " + (yearly ? "" : "MONTH, ") + "AFFILIATION_1, AFFILIATION_2, MACHINE.NAME");
                } else if (type == 2) {
                    rs = stmt.executeQuery("select YEAR(DATE) as YEAR, " + (yearly ? "" : " MONTH(DATE) as MONTH, ") + "MACHINE.NAME as NAME, count(*) as C, sum(WISHED_SPECTRUM like '%13C%' or WISHED_SPECTRUM like '%variable temperature%' or WISHED_SPECTRUM like '%ID sel. NOE%' or WISHED_SPECTRUM like '%solvent suppression%' or WISHED_SPECTRUM like '%standard spectrum%') as 1_D, sum(WISHED_SPECTRUM like '%H,H-COSY%' or WISHED_SPECTRUM like '%NOESY%' or WISHED_SPECTRUM like '%HMQC%' or WISHED_SPECTRUM like '%HMBC%') as 2_D, sum(OTHER_WISHED_SPECTRUM!='') as SPECIAL, sum(OTHER_NUCLEI!='') as HETERO, sum(PROCESS='self') as SELF, sum(PROCESS='robot') as ROBOT, sum(PROCESS='worker') as OPERATOR from (SAMPLE join TURBINE_USER using (USER_ID)) join MACHINE on MACHINE.MACHINE_ID=SAMPLE.MACHINE group by YEAR, " + (yearly ? "" : "MONTH, ") + "MACHINE.NAME");
                }
                JasperPrint jasperPrint = JasperFillManager.fillReport(jasperReport, parameters, new JRResultSetDataSource(rs));
                JasperExportManager.exportReportToPdfStream(jasperPrint, outstream);
                dbconn.close();
            } else if (action.equals("exportcmlbyinchi")) {
                res.setContentType("text/xml");
                out = res.getWriter();
                String inchi = req.getParameter("inchi");
                String spectrumtype = req.getParameter("spectrumtype");
                Criteria crit = new Criteria();
                crit.add(DBCanonicalNamePeer.NAME, inchi);
                crit.addJoin(DBCanonicalNamePeer.MOLECULE_ID, DBSpectrumPeer.MOLECULE_ID);
                crit.addJoin(DBSpectrumPeer.SPECTRUM_TYPE_ID, DBSpectrumTypePeer.SPECTRUM_TYPE_ID);
                crit.add(DBSpectrumTypePeer.NAME, spectrumtype);
                try {
                    GeneralUtils.logToSql(crit.toString(), null);
                } catch (Exception ex) {
                }
                Vector spectra = DBSpectrumPeer.doSelect(crit);
                if (spectra.size() == 0) {
                    out.write("No such molecule or spectrum");
                } else {
                    Element cmlElement = new Element("cml");
                    cmlElement.addAttribute(new Attribute("convention", "nmrshiftdb-convention"));
                    cmlElement.setNamespaceURI("http://www.xml-cml.org/schema");
                    Element parent = ((DBSpectrum) spectra.get(0)).getDBMolecule().getCML(1);
                    nu.xom.Node cmldoc = parent.getChild(0);
                    ((Element) cmldoc).setNamespaceURI("http://www.xml-cml.org/schema");
                    parent.removeChildren();
                    cmlElement.appendChild(cmldoc);
                    for (int k = 0; k < spectra.size(); k++) {
                        Element parentspec = ((DBSpectrum) spectra.get(k)).getCmlSpect();
                        Node spectrumel = parentspec.getChild(0);
                        parentspec.removeChildren();
                        cmlElement.appendChild(spectrumel);
                        ((Element) spectrumel).setNamespaceURI("http://www.xml-cml.org/schema");
                    }
                    out.write(cmlElement.toXML());
                }
            } else if (action.equals("namelist")) {
                res.setContentType("application/zip");
                outstream = res.getOutputStream();
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                ZipOutputStream zipout = new ZipOutputStream(baos);
                Criteria crit = new Criteria();
                crit.addJoin(DBMoleculePeer.MOLECULE_ID, DBSpectrumPeer.MOLECULE_ID);
                crit.add(DBSpectrumPeer.REVIEW_FLAG, "true");
                Vector v = DBMoleculePeer.doSelect(crit);
                for (int i = 0; i < v.size(); i++) {
                    if (i % 500 == 0) {
                        if (i != 0) {
                            zipout.write(new String("<p>The list is continued <a href=\"nmrshiftdb.names." + i + ".html\">here</a></p></body></html>").getBytes());
                            zipout.closeEntry();
                        }
                        zipout.putNextEntry(new ZipEntry("nmrshiftdb.names." + i + ".html"));
                        zipout.write(new String("<html><body><h1>This is a list of strcutures in <a href=\"http://www.nmrshiftdb.org\">NMRShiftDB</a>, starting at " + i + ", Its main purpose is to be found by search engines</h1>").getBytes());
                    }
                    DBMolecule mol = (DBMolecule) v.get(i);
                    zipout.write(new String("<p><a href=\"" + mol.getEasylink(req) + "\">").getBytes());
                    Vector cannames = mol.getDBCanonicalNames();
                    for (int k = 0; k < cannames.size(); k++) {
                        zipout.write(new String(((DBCanonicalName) cannames.get(k)).getName() + " ").getBytes());
                    }
                    Vector chemnames = mol.getDBChemicalNames();
                    for (int k = 0; k < chemnames.size(); k++) {
                        zipout.write(new String(((DBChemicalName) chemnames.get(k)).getName() + " ").getBytes());
                    }
                    zipout.write(new String("</a>. Information we have got: NMR spectra").getBytes());
                    Vector spectra = mol.selectSpectra();
                    for (int k = 0; k < spectra.size(); k++) {
                        zipout.write(new String(((DBSpectrum) spectra.get(k)).getDBSpectrumType().getName() + ", ").getBytes());
                    }
                    if (mol.hasAny3d()) zipout.write(new String("3D coordinates, ").getBytes());
                    zipout.write(new String("File formats: CML, mol, png, jpeg").getBytes());
                    zipout.write(new String("</p>").getBytes());
                }
                zipout.write(new String("</body></html>").getBytes());
                zipout.closeEntry();
                zipout.close();
                InputStream is = new ByteArrayInputStream(baos.toByteArray());
                byte[] buf = new byte[32 * 1024];
                int nRead = 0;
                while ((nRead = is.read(buf)) != -1) {
                    outstream.write(buf, 0, nRead);
                }
            } else if (action.equals("predictor")) {
                if (req.getParameter("symbol") == null) {
                    res.setContentType("text/plain");
                    out = res.getWriter();
                    out.write("please give the symbol to create the predictor for in the request with symbol=X (e. g. symbol=C");
                }
                res.setContentType("application/zip");
                outstream = res.getOutputStream();
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                ZipOutputStream zipout = new ZipOutputStream(baos);
                String filename = "org/openscience/nmrshiftdb/PredictionTool.class";
                zipout.putNextEntry(new ZipEntry(filename));
                JarInputStream jip = new JarInputStream(new FileInputStream(ServletUtils.expandRelative(getServletConfig(), "/WEB-INF/lib/nmrshiftdb-lib.jar")));
                JarEntry entry = jip.getNextJarEntry();
                while (entry.getName().indexOf("PredictionTool.class") == -1) {
                    entry = jip.getNextJarEntry();
                }
                for (int i = 0; i < entry.getSize(); i++) {
                    zipout.write(jip.read());
                }
                zipout.closeEntry();
                zipout.putNextEntry(new ZipEntry("nmrshiftdb.csv"));
                int i = 0;
                org.apache.turbine.util.db.pool.DBConnection conn = TurbineDB.getConnection();
                HashMap mapsmap = new HashMap();
                while (true) {
                    Statement stmt = conn.createStatement();
                    ResultSet rs = stmt.executeQuery("select HOSE_CODE, VALUE, SYMBOL from HOSE_CODES where CONDITION_TYPE='m' and WITH_RINGS=0 and SYMBOL='" + req.getParameter("symbol") + "' limit " + (i * 1000) + ", 1000");
                    int m = 0;
                    while (rs.next()) {
                        String code = rs.getString(1);
                        Double value = new Double(rs.getString(2));
                        String symbol = rs.getString(3);
                        if (mapsmap.get(symbol) == null) {
                            mapsmap.put(symbol, new HashMap());
                        }
                        for (int spheres = 6; spheres > 0; spheres--) {
                            StringBuffer hoseCodeBuffer = new StringBuffer();
                            StringTokenizer st = new StringTokenizer(code, "()/");
                            for (int k = 0; k < spheres; k++) {
                                if (st.hasMoreTokens()) {
                                    String partcode = st.nextToken();
                                    hoseCodeBuffer.append(partcode);
                                }
                                if (k == 0) {
                                    hoseCodeBuffer.append("(");
                                } else if (k == 3) {
                                    hoseCodeBuffer.append(")");
                                } else {
                                    hoseCodeBuffer.append("/");
                                }
                            }
                            String hoseCode = hoseCodeBuffer.toString();
                            if (((HashMap) mapsmap.get(symbol)).get(hoseCode) == null) {
                                ((HashMap) mapsmap.get(symbol)).put(hoseCode, new ArrayList());
                            }
                            ((ArrayList) ((HashMap) mapsmap.get(symbol)).get(hoseCode)).add(value);
                        }
                        m++;
                    }
                    i++;
                    stmt.close();
                    if (m == 0) break;
                }
                Set keySet = mapsmap.keySet();
                Iterator it = keySet.iterator();
                while (it.hasNext()) {
                    String symbol = (String) it.next();
                    HashMap hosemap = ((HashMap) mapsmap.get(symbol));
                    Set keySet2 = hosemap.keySet();
                    Iterator it2 = keySet2.iterator();
                    while (it2.hasNext()) {
                        String hoseCode = (String) it2.next();
                        ArrayList list = ((ArrayList) hosemap.get(hoseCode));
                        double[] values = new double[list.size()];
                        for (int k = 0; k < list.size(); k++) {
                            values[k] = ((Double) list.get(k)).doubleValue();
                        }
                        zipout.write(new String(symbol + "|" + hoseCode + "|" + Statistics.minimum(values) + "|" + Statistics.average(values) + "|" + Statistics.maximum(values) + "\r\n").getBytes());
                    }
                }
                zipout.closeEntry();
                zipout.close();
                InputStream is = new ByteArrayInputStream(baos.toByteArray());
                byte[] buf = new byte[32 * 1024];
                int nRead = 0;
                i = 0;
                while ((nRead = is.read(buf)) != -1) {
                    outstream.write(buf, 0, nRead);
                }
            } else if (action.equals("exportspec") || action.equals("exportmol")) {
                if (spectrumId > -1) spectrum = DBSpectrumPeer.retrieveByPK(new NumberKey(spectrumId)); else spectrum = DBSpectrumPeer.retrieveByPK(new NumberKey(req.getParameter("spectrumid")));
                export = new Export(spectrum);
            } else if (action.equals("exportmdl")) {
                res.setContentType("text/plain");
                outstream = res.getOutputStream();
                DBMolecule mol = DBMoleculePeer.retrieveByPK(new NumberKey(req.getParameter("moleculeid")));
                outstream.write(mol.getStructureFile(Integer.parseInt(req.getParameter("coordsetid")), false).getBytes());
            } else if (action.equals("exportlastinputs")) {
                format = action;
            } else if (action.equals("printpredict")) {
                res.setContentType("text/html");
                out = res.getWriter();
                HttpSession session = req.getSession();
                VelocityContext context = PredictPortlet.getContext(session, true, true, new StringBuffer(), getServletConfig(), req, true);
                StringWriter w = new StringWriter();
                Velocity.mergeTemplate("predictprint.vm", "ISO-8859-1", context, w);
                out.println(w.toString());
            } else {
                res.setContentType("text/html");
                out = res.getWriter();
                out.println("No valid action");
            }
            if (format == null) format = "";
            if (format.equals("pdf") || format.equals("rtf")) {
                res.setContentType("application/" + format);
                out = res.getWriter();
            }
            if (format.equals("docbook")) {
                res.setContentType("application/zip");
                outstream = res.getOutputStream();
            }
            if (format.equals("svg")) {
                res.setContentType("image/x-svg");
                out = res.getWriter();
            }
            if (format.equals("tiff")) {
                res.setContentType("image/tiff");
                outstream = res.getOutputStream();
            }
            if (format.equals("jpeg")) {
                res.setContentType("image/jpeg");
                outstream = res.getOutputStream();
            }
            if (format.equals("png")) {
                res.setContentType("image/png");
                outstream = res.getOutputStream();
            }
            if (format.equals("mdl") || format.equals("txt") || format.equals("cml") || format.equals("cmlboth") || format.indexOf("exsection") == 0) {
                res.setContentType("text/plain");
                out = res.getWriter();
            }
            if (format.equals("simplehtml") || format.equals("exportlastinputs")) {
                res.setContentType("text/html");
                out = res.getWriter();
            }
            if (action.equals("exportlastinputs")) {
                int numbertoexport = 4;
                if (req.getParameter("numbertoexport") != null) {
                    try {
                        numbertoexport = Integer.parseInt(req.getParameter("numbertoexport"));
                        if (numbertoexport < 1 || numbertoexport > 20) throw new NumberFormatException("Number to small/large");
                    } catch (NumberFormatException ex) {
                        out.println("The parameter <code>numbertoexport</code>must be an integer from 1 to 20");
                    }
                }
                NmrshiftdbUser user = null;
                try {
                    user = NmrshiftdbUserPeer.getByName(req.getParameter("username"));
                } catch (NmrshiftdbException ex) {
                    out.println("Seems <code>username</code> is not OK: " + ex.getMessage());
                }
                if (user != null) {
                    List l = NmrshiftdbUserPeer.executeQuery("SELECT LAST_DOWNLOAD_DATE FROM TURBINE_USER  where LOGIN_NAME=\"" + user.getUserName() + "\";");
                    Date lastDownloadDate = ((Record) l.get(0)).getValue(1).asDate();
                    if (((new Date().getTime() - lastDownloadDate.getTime()) / 3600000) < 24) {
                        out.println("Your last download was at " + lastDownloadDate + ". You may download your last inputs only once a day. Sorry for this, but we need to be carefull with resources. If you want to put your last inputs on your homepage best use some sort of cache (e. g. use wget for downlaod with crond and link to this static resource))!");
                    } else {
                        NmrshiftdbUserPeer.executeStatement("UPDATE TURBINE_USER SET LAST_DOWNLOAD_DATE=NOW() where LOGIN_NAME=\"" + user.getUserName() + "\";");
                        Vector<String> parameters = new Vector<String>();
                        String query = "select distinct MOLECULE.MOLECULE_ID from MOLECULE, SPECTRUM where SPECTRUM.MOLECULE_ID = MOLECULE.MOLECULE_ID and SPECTRUM.REVIEW_FLAG =\"true\" and SPECTRUM.USER_ID=" + user.getUserId() + " order by MOLECULE.DATE desc;";
                        l = NmrshiftdbUserPeer.executeQuery(query);
                        String url = javax.servlet.http.HttpUtils.getRequestURL(req).toString();
                        url = url.substring(0, url.length() - 17);
                        for (int i = 0; i < numbertoexport; i++) {
                            if (i == l.size()) break;
                            DBMolecule mol = DBMoleculePeer.retrieveByPK(new NumberKey(((Record) l.get(i)).getValue(1).asInt()));
                            parameters.add(new String("<a href=\"" + url + "/portal/pane0/Results?nmrshiftdbaction=showDetailsFromHome&molNumber=" + mol.getMoleculeId() + "\"><img src=\"" + javax.servlet.http.HttpUtils.getRequestURL(req).toString() + "?nmrshiftdbaction=exportmol&spectrumid=" + ((DBSpectrum) mol.getDBSpectrums().get(0)).getSpectrumId() + "&format=jpeg&size=150x150&backcolor=12632256\"></a>"));
                        }
                        VelocityContext context = new VelocityContext();
                        context.put("results", parameters);
                        StringWriter w = new StringWriter();
                        Velocity.mergeTemplate("lateststructures.vm", "ISO-8859-1", context, w);
                        out.println(w.toString());
                    }
                }
            }
            if (action.equals("exportspec")) {
                if (format.equals("txt")) {
                    String lastsearchtype = req.getParameter("lastsearchtype");
                    if (lastsearchtype.equals(NmrshiftdbConstants.TOTALSPECTRUM) || lastsearchtype.equals(NmrshiftdbConstants.SUBSPECTRUM)) {
                        List l = ParseUtils.parseSpectrumFromSpecFile(req.getParameter("lastsearchvalues"));
                        spectrum.initSimilarity(l, lastsearchtype.equals(NmrshiftdbConstants.SUBSPECTRUM));
                    }
                    Vector v = spectrum.getOptions();
                    DBMolecule mol = spectrum.getDBMolecule();
                    out.print(mol.getChemicalNamesAsOneString(false) + mol.getMolecularFormula(false) + "; " + mol.getMolecularWeight() + " Dalton\n\r");
                    out.print("\n\rAtom\t");
                    if (spectrum.getDBSpectrumType().getElementSymbol() == ("H")) out.print("Mult.\t");
                    out.print("Meas.");
                    if (lastsearchtype.equals(NmrshiftdbConstants.TOTALSPECTRUM) || lastsearchtype.equals(NmrshiftdbConstants.SUBSPECTRUM)) {
                        out.print("\tInput\tDiff");
                    }
                    out.print("\n\r");
                    out.print("No.\t");
                    if (spectrum.getDBSpectrumType().getElementSymbol() == ("H")) out.print("\t");
                    out.print("Shift");
                    if (lastsearchtype.equals(NmrshiftdbConstants.TOTALSPECTRUM) || lastsearchtype.equals(NmrshiftdbConstants.SUBSPECTRUM)) {
                        out.print("\tShift\tM-I");
                    }
                    out.print("\n\r");
                    for (int i = 0; i < v.size(); i++) {
                        out.print(((ValuesForVelocityBean) v.get(i)).getDisplayText() + "\t" + ((ValuesForVelocityBean) v.get(i)).getRange());
                        if (lastsearchtype.equals(NmrshiftdbConstants.TOTALSPECTRUM) || lastsearchtype.equals(NmrshiftdbConstants.SUBSPECTRUM)) {
                            out.print("\t" + ((ValuesForVelocityBean) v.get(i)).getNameForElements() + "\t" + ((ValuesForVelocityBean) v.get(i)).getDelta());
                        }
                        out.print("\n\r");
                    }
                }
                if (format.equals("simplehtml")) {
                    String i1 = export.getImage(false, "jpeg", ServletUtils.expandRelative(this.getServletConfig(), "/nmrshiftdbhtml") + "/tmp/" + System.currentTimeMillis(), true);
                    export.pictures[0] = new File(i1).getName();
                    String i2 = export.getImage(true, "jpeg", ServletUtils.expandRelative(this.getServletConfig(), "/nmrshiftdbhtml") + "/tmp/" + System.currentTimeMillis(), true);
                    export.pictures[1] = new File(i2).getName();
                    String docbook = export.getHtml();
                    out.print(docbook);
                }
                if (format.equals("pdf") || format.equals("rtf")) {
                    String svgSpec = export.getSpecSvg(400, 200);
                    String svgspecfile = relativepath + "/tmp/" + System.currentTimeMillis() + "s.svg";
                    new FileOutputStream(svgspecfile).write(svgSpec.getBytes());
                    export.pictures[1] = svgspecfile;
                    String molSvg = export.getMolSvg(true);
                    String svgmolfile = relativepath + "/tmp/" + System.currentTimeMillis() + "m.svg";
                    new FileOutputStream(svgmolfile).write(molSvg.getBytes());
                    export.pictures[0] = svgmolfile;
                    String docbook = export.getDocbook("pdf", "SVG");
                    TransformerFactory tFactory = TransformerFactory.newInstance();
                    Transformer transformer = tFactory.newTransformer(new StreamSource("file:" + GeneralUtils.getNmrshiftdbProperty("docbookxslpath", getServletConfig()) + "/fo/docbook.xsl"));
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    transformer.transform(new StreamSource(new StringReader(docbook)), new StreamResult(baos));
                    FopFactory fopFactory = FopFactory.newInstance();
                    FOUserAgent foUserAgent = fopFactory.newFOUserAgent();
                    OutputStream out2 = new ByteArrayOutputStream();
                    Fop fop = fopFactory.newFop(format.equals("rtf") ? MimeConstants.MIME_RTF : MimeConstants.MIME_PDF, foUserAgent, out2);
                    TransformerFactory factory = TransformerFactory.newInstance();
                    transformer = factory.newTransformer();
                    Source src = new StreamSource(new StringReader(baos.toString()));
                    Result res2 = new SAXResult(fop.getDefaultHandler());
                    transformer.transform(src, res2);
                    out.print(out2.toString());
                }
                if (format.equals("docbook")) {
                    String i1 = relativepath + "/tmp/" + System.currentTimeMillis() + ".svg";
                    new FileOutputStream(i1).write(export.getSpecSvg(300, 200).getBytes());
                    export.pictures[0] = new File(i1).getName();
                    String i2 = relativepath + "/tmp/" + System.currentTimeMillis() + ".svg";
                    new FileOutputStream(i2).write(export.getMolSvg(true).getBytes());
                    export.pictures[1] = new File(i2).getName();
                    String docbook = export.getDocbook("pdf", "SVG");
                    String docbookfile = relativepath + "/tmp/" + System.currentTimeMillis() + ".xml";
                    new FileOutputStream(docbookfile).write(docbook.getBytes());
                    ByteArrayOutputStream baos = export.makeZip(new String[] { docbookfile, i1, i2 });
                    outstream.write(baos.toByteArray());
                }
                if (format.equals("svg")) {
                    out.print(export.getSpecSvg(400, 200));
                }
                if (format.equals("tiff") || format.equals("jpeg") || format.equals("png")) {
                    InputStream is = new FileInputStream(export.getImage(false, format, relativepath + "/tmp/" + System.currentTimeMillis(), true));
                    byte[] buf = new byte[32 * 1024];
                    int nRead = 0;
                    while ((nRead = is.read(buf)) != -1) {
                        outstream.write(buf, 0, nRead);
                    }
                }
                if (format.equals("cml")) {
                    out.print(spectrum.getCmlSpect().toXML());
                }
                if (format.equals("cmlboth")) {
                    Element cmlElement = new Element("cml");
                    cmlElement.addAttribute(new Attribute("convention", "nmrshiftdb-convention"));
                    cmlElement.setNamespaceURI("http://www.xml-cml.org/schema");
                    Element parent = spectrum.getDBMolecule().getCML(1, spectrum.getDBSpectrumType().getName().equals("1H"));
                    nu.xom.Node cmldoc = parent.getChild(0);
                    ((Element) cmldoc).setNamespaceURI("http://www.xml-cml.org/schema");
                    parent.removeChildren();
                    cmlElement.appendChild(cmldoc);
                    Element parentspec = spectrum.getCmlSpect();
                    Node spectrumel = parentspec.getChild(0);
                    parentspec.removeChildren();
                    cmlElement.appendChild(spectrumel);
                    ((Element) spectrumel).setNamespaceURI("http://www.xml-cml.org/schema");
                    out.write(cmlElement.toXML());
                }
                if (format.indexOf("exsection") == 0) {
                    StringTokenizer st = new StringTokenizer(format, "-");
                    st.nextToken();
                    String template = st.nextToken();
                    Criteria crit = new Criteria();
                    crit.add(DBSpectrumPeer.USER_ID, spectrum.getUserId());
                    Vector v = spectrum.getDBMolecule().getDBSpectrums(crit);
                    VelocityContext context = new VelocityContext();
                    context.put("spectra", v);
                    context.put("molecule", spectrum.getDBMolecule());
                    StringWriter w = new StringWriter();
                    Velocity.mergeTemplate("exporttemplates/" + template, "ISO-8859-1", context, w);
                    out.write(w.toString());
                }
            }
            if (action.equals("exportmol")) {
                int width = -1;
                int height = -1;
                if (req.getParameter("size") != null) {
                    StringTokenizer st = new StringTokenizer(req.getParameter("size"), "x");
                    width = Integer.parseInt(st.nextToken());
                    height = Integer.parseInt(st.nextToken());
                }
                boolean shownumbers = true;
                if (req.getParameter("shownumbers") != null && req.getParameter("shownumbers").equals("false")) {
                    shownumbers = false;
                }
                if (req.getParameter("backcolor") != null) {
                    export.backColor = new Color(Integer.parseInt(req.getParameter("backcolor")));
                }
                if (req.getParameter("markatom") != null) {
                    export.selected = Integer.parseInt(req.getParameter("markatom")) - 1;
                }
                if (format.equals("svg")) {
                    out.print(export.getMolSvg(true));
                }
                if (format.equals("tiff") || format.equals("jpeg") || format.equals("png")) {
                    InputStream is = new FileInputStream(export.getImage(true, format, relativepath + "/tmp/" + System.currentTimeMillis(), width, height, shownumbers, null));
                    byte[] buf = new byte[32 * 1024];
                    int nRead = 0;
                    while ((nRead = is.read(buf)) != -1) {
                        outstream.write(buf, 0, nRead);
                    }
                }
                if (format.equals("mdl")) {
                    out.println(spectrum.getDBMolecule().getStructureFile(1, false));
                }
                if (format.equals("cml")) {
                    out.println(spectrum.getDBMolecule().getCMLString(1));
                }
            }
            if (out != null) out.flush(); else outstream.flush();
        } catch (Exception ex) {
            ex.printStackTrace();
            out.print(GeneralUtils.logError(ex, "NmrshiftdbServlet", null, true));
            out.flush();
        }
    }
```

## P082

**A**

```java
public boolean excuteBackup(String backupOrginlDrctry, String targetFileNm, String archiveFormat) throws JobExecutionException {
        File targetFile = new File(targetFileNm);
        File srcFile = new File(backupOrginlDrctry);
        if (!srcFile.exists()) {
            log.error("백업원본디렉토리[" + srcFile.getAbsolutePath() + "]가 존재하지 않습니다.");
            throw new JobExecutionException("백업원본디렉토리[" + srcFile.getAbsolutePath() + "]가 존재하지 않습니다.");
        }
        if (srcFile.isFile()) {
            log.error("백업원본디렉토리[" + srcFile.getAbsolutePath() + "]가 파일입니다. 디렉토리명을 지정해야 합니다. ");
            throw new JobExecutionException("백업원본디렉토리[" + srcFile.getAbsolutePath() + "]가 파일입니다. 디렉토리명을 지정해야 합니다. ");
        }
        boolean result = false;
        FileInputStream finput = null;
        FileOutputStream fosOutput = null;
        ArchiveOutputStream aosOutput = null;
        ArchiveEntry entry = null;
        try {
            log.debug("charter set : " + Charset.defaultCharset().name());
            fosOutput = new FileOutputStream(targetFile);
            aosOutput = new ArchiveStreamFactory().createArchiveOutputStream(archiveFormat, fosOutput);
            if (ArchiveStreamFactory.TAR.equals(archiveFormat)) {
                ((TarArchiveOutputStream) aosOutput).setLongFileMode(TarArchiveOutputStream.LONGFILE_GNU);
            }
            File[] fileArr = srcFile.listFiles();
            ArrayList list = EgovFileTool.getSubFilesByAll(fileArr);
            for (int i = 0; i < list.size(); i++) {
                File sfile = new File((String) list.get(i));
                finput = new FileInputStream(sfile);
                if (ArchiveStreamFactory.TAR.equals(archiveFormat)) {
                    entry = new TarArchiveEntry(sfile, new String(sfile.getAbsolutePath().getBytes(Charset.defaultCharset().name()), "8859_1"));
                    ((TarArchiveEntry) entry).setSize(sfile.length());
                } else {
                    entry = new ZipArchiveEntry(sfile.getAbsolutePath());
                    ((ZipArchiveEntry) entry).setSize(sfile.length());
                }
                aosOutput.putArchiveEntry(entry);
                IOUtils.copy(finput, aosOutput);
                aosOutput.closeArchiveEntry();
                finput.close();
                result = true;
            }
            aosOutput.close();
        } catch (Exception e) {
            log.error("백업화일생성중 에러가 발생했습니다. 에러 : " + e.getMessage());
            log.debug(e);
            result = false;
            throw new JobExecutionException("백업화일생성중 에러가 발생했습니다.", e);
        } finally {
            try {
                if (finput != null) finput.close();
            } catch (Exception e2) {
                log.error("IGNORE:", e2);
            }
            try {
                if (aosOutput != null) aosOutput.close();
            } catch (Exception e2) {
                log.error("IGNORE:", e2);
            }
            try {
                if (fosOutput != null) fosOutput.close();
            } catch (Exception e2) {
                log.error("IGNORE:", e2);
            }
            try {
                if (result == false) targetFile.delete();
            } catch (Exception e2) {
                log.error("IGNORE:", e2);
            }
        }
        return result;
    }
```

**B**

```java
private void prepareQueryResultData(ZipEntryRef zer, String nodeDir, String reportDir, Set<ZipEntryRef> statusZers) throws Exception {
        String jobDir = nodeDir + File.separator + "job_" + zer.getUri();
        if (!WorkDirectory.isWorkingDirectoryValid(jobDir)) {
            throw new Exception("Cannot acces to " + jobDir);
        }
        File f = new File(jobDir + File.separator + "result.xml");
        if (!f.exists() || !f.isFile() || !f.canRead()) {
            throw new Exception("Cannot acces to result file " + f.getAbsolutePath());
        }
        String fcopyName = reportDir + File.separator + zer.getName() + ".xml";
        BufferedInputStream bis = new BufferedInputStream(new FileInputStream(f));
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(fcopyName));
        IOUtils.copy(bis, bos);
        bis.close();
        bos.close();
        zer.setUri(fcopyName);
        f = new File(jobDir + File.separator + "status.xml");
        if (!f.exists() || !f.isFile() || !f.canRead()) {
            throw new Exception("Cannot acces to status file " + f.getAbsolutePath());
        }
        fcopyName = reportDir + File.separator + zer.getName() + "_status.xml";
        bis = new BufferedInputStream(new FileInputStream(f));
        bos = new BufferedOutputStream(new FileOutputStream(fcopyName));
        IOUtils.copy(bis, bos);
        bis.close();
        bos.close();
        statusZers.add(new ZipEntryRef(ZipEntryRef.SINGLE_FILE, zer.getName(), fcopyName, ZipEntryRef.WITH_REL));
    }
```

## P083

**A**

```java
private static void addFile(File file, TarArchiveOutputStream taos) throws IOException {
        String filename = null;
        filename = file.getName();
        TarArchiveEntry tae = new TarArchiveEntry(filename);
        tae.setSize(file.length());
        taos.putArchiveEntry(tae);
        FileInputStream fis = new FileInputStream(file);
        IOUtils.copy(fis, taos);
        taos.closeArchiveEntry();
    }
```

**B**

```java
public static void copyFile(File dst, File src, boolean append) throws FileNotFoundException, IOException {
        dst.createNewFile();
        FileChannel in = new FileInputStream(src).getChannel();
        FileChannel out = new FileOutputStream(dst).getChannel();
        long startAt = 0;
        if (append) startAt = out.size();
        in.transferTo(startAt, in.size(), out);
        out.close();
        in.close();
    }
```

## P084

**A**

```java
private String generateCode(String seed) {
        try {
            Security.addProvider(new FNVProvider());
            MessageDigest digest = MessageDigest.getInstance("FNV-1a");
            digest.update((seed + UUID.randomUUID().toString()).getBytes());
            byte[] hash1 = digest.digest();
            String sHash1 = "m" + (new String(LibraryBase64.encode(hash1))).replaceAll("=", "").replaceAll("-", "_");
            return sHash1;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return "";
    }
```

**B**

```java
protected String insertCommand(String command) throws ServletException {
        String digest;
        try {
            MessageDigest md = MessageDigest.getInstance(m_messagedigest_algorithm);
            md.update(command.getBytes());
            byte bytes[] = new byte[20];
            m_random.nextBytes(bytes);
            md.update(bytes);
            digest = bytesToHex(md.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new ServletException("NoSuchAlgorithmException while " + "attempting to generate graph ID: " + e);
        }
        String id = System.currentTimeMillis() + "-" + digest;
        m_map.put(id, command);
        return id;
    }
```

## P085

**A**

```java
public static void copyFile(String source, String dest) throws IOException {
        FileChannel in = null, out = null;
        try {
            in = new FileInputStream(new File(source)).getChannel();
            out = new FileOutputStream(new File(dest)).getChannel();
            in.transferTo(0, in.size(), out);
        } finally {
            if (in != null) in.close();
            if (out != null) out.close();
        }
    }
```

**B**

```java
public static void copyFile(final File sourceFile, final File destFile) throws IOException {
        if (destFile.getParentFile() != null && !destFile.getParentFile().mkdirs()) {
            LOG.error("GeneralHelper.copyFile(): Cannot create parent directories from " + destFile);
        }
        FileInputStream fIn = null;
        FileOutputStream fOut = null;
        FileChannel source = null;
        FileChannel destination = null;
        try {
            fIn = new FileInputStream(sourceFile);
            source = fIn.getChannel();
            fOut = new FileOutputStream(destFile);
            destination = fOut.getChannel();
            long transfered = 0;
            final long bytes = source.size();
            while (transfered < bytes) {
                transfered += destination.transferFrom(source, 0, source.size());
                destination.position(transfered);
            }
        } finally {
            if (source != null) {
                source.close();
            } else if (fIn != null) {
                fIn.close();
            }
            if (destination != null) {
                destination.close();
            } else if (fOut != null) {
                fOut.close();
            }
        }
    }
```

## P086

**A**

```java
public static String md5Encode16(String s) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            md.update(s.getBytes("utf-8"));
            byte b[] = md.digest();
            int i;
            StringBuilder buf = new StringBuilder("");
            for (int offset = 0; offset < b.length; offset++) {
                i = b[offset];
                if (i < 0) i += 256;
                if (i < 16) buf.append("0");
                buf.append(Integer.toHexString(i));
            }
            return buf.toString().substring(8, 24);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalArgumentException(e);
        } catch (UnsupportedEncodingException e) {
            throw new IllegalArgumentException(e);
        }
    }
```

**B**

```java
public String encode(String plain) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            md.update(plain.getBytes());
            byte b[] = md.digest();
            int i;
            StringBuffer buf = new StringBuffer("");
            for (int offset = 0; offset < b.length; offset++) {
                i = b[offset];
                if (i < 0) i += 256;
                if (i < 16) buf.append("0");
                buf.append(Integer.toHexString(i));
            }
            return buf.toString();
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        }
        return null;
    }
```

## P087

**A**

```java
private String fetchHTML(String s) {
        String str;
        StringBuffer sb = new StringBuffer();
        try {
            URL url = new URL(s);
            InputStream is = url.openStream();
            InputStreamReader isr = new InputStreamReader(is);
            BufferedReader br = new BufferedReader(isr);
            while ((str = br.readLine()) != null) {
                sb.append(str);
            }
        } catch (MalformedURLException e) {
        } catch (IOException e) {
        }
        return sb.toString();
    }
```

**B**

```java
private String getPayLoadWithCookie(String url) {
        StringBuffer sb = new StringBuffer();
        if (this.cookie != null) {
            try {
                Log.debug("Requesting url ==> " + url);
                URLConnection con = new URL(url).openConnection();
                con.setDoOutput(true);
                con.addRequestProperty("Cookie", this.cookie);
                BufferedReader br = new BufferedReader(new InputStreamReader(con.getInputStream()));
                String line = "";
                while ((line = br.readLine()) != null) {
                    sb.append(line + "\n");
                }
            } catch (MalformedURLException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return sb.toString();
    }
```

## P088

**A**

```java
public void downloadFile(OutputStream os, int fileId) throws IOException, SQLException {
        Connection conn = null;
        try {
            conn = ds.getConnection();
            Guard.checkConnectionNotNull(conn);
            PreparedStatement ps = conn.prepareStatement("select * from FILE_BODIES where file_id=?");
            ps.setInt(1, fileId);
            ResultSet rs = ps.executeQuery();
            if (!rs.next()) {
                throw new FileNotFoundException("File with id=" + fileId + " not found!");
            }
            Blob blob = rs.getBlob("data");
            InputStream is = blob.getBinaryStream();
            IOUtils.copyLarge(is, os);
        } finally {
            JdbcDaoHelper.safeClose(conn, log);
        }
    }
```

**B**

```java
public void save(String oid, String key, Serializable obj) throws PersisterException {
        String lock = getLock(oid);
        if (lock == null) {
            throw new PersisterException("Object does not exist: OID = " + oid);
        } else if (!NULL.equals(lock) && (!lock.equals(key))) {
            throw new PersisterException("The object is currently locked with another key: OID = " + oid + ", LOCK = " + lock + ", KEY = " + key);
        }
        Connection conn = null;
        PreparedStatement ps = null;
        try {
            byte[] data = serialize(obj);
            conn = _ds.getConnection();
            conn.setAutoCommit(true);
            ps = conn.prepareStatement("update " + _table_name + " set " + _data_col + " = ?, " + _ts_col + " = ? where " + _oid_col + " = ?");
            ps.setBinaryStream(1, new ByteArrayInputStream(data), data.length);
            ps.setLong(2, System.currentTimeMillis());
            ps.setString(3, oid);
            ps.executeUpdate();
        } catch (Throwable th) {
            if (conn != null) {
                try {
                    conn.rollback();
                } catch (Throwable th2) {
                }
            }
            throw new PersisterException("Failed to save object: OID = " + oid, th);
        } finally {
            if (ps != null) {
                try {
                    ps.close();
                } catch (Throwable th) {
                }
            }
            if (conn != null) {
                try {
                    conn.close();
                } catch (Throwable th) {
                }
            }
        }
    }
```

## P089

**A**

```java
public int process(ProcessorContext context) throws InterruptedException, ProcessorException {
        logger.info("JAISaveTask:process");
        final RenderedOp im = (RenderedOp) context.get("RenderedOp");
        final String path = "s3://s3.amazonaws.com/rssfetch/" + (new Guid());
        final PNGEncodeParam.RGB encPar = new PNGEncodeParam.RGB();
        encPar.setTransparentRGB(new int[] { 0, 0, 0 });
        File tmpFile = null;
        try {
            tmpFile = File.createTempFile("thmb", ".png");
            OutputStream out = new FileOutputStream(tmpFile);
            final ParameterBlock pb = (new ParameterBlock()).addSource(im).add(out).add("png").add(encPar);
            JAI.create("encode", pb, null);
            out.flush();
            out.close();
            FileInputStream in = new FileInputStream(tmpFile);
            final XFile xfile = new XFile(path);
            final XFileOutputStream xout = new XFileOutputStream(xfile);
            final com.luzan.common.nfs.s3.XFileExtensionAccessor xfa = ((com.luzan.common.nfs.s3.XFileExtensionAccessor) xfile.getExtensionAccessor());
            if (xfa != null) {
                xfa.setMimeType("image/png");
                xfa.setContentLength(tmpFile.length());
            }
            IOUtils.copy(in, xout);
            xout.flush();
            xout.close();
            in.close();
            context.put("outputPath", path);
        } catch (IOException e) {
            logger.error(e);
            throw new ProcessorException(e);
        } catch (Throwable e) {
            logger.error(e);
            throw new ProcessorException(e);
        } finally {
            if (tmpFile != null && tmpFile.exists()) {
                tmpFile.delete();
            }
        }
        return TaskState.STATE_MO_START + TaskState.STATE_ENCODE;
    }
```

**B**

```java
private String save(UploadedFile imageFile) {
        try {
            File saveFld = new File(imageFolder + File.separator + userDisplay.getUser().getUsername());
            if (!saveFld.exists()) {
                if (!saveFld.mkdir()) {
                    logger.info("Unable to create folder: " + saveFld.getAbsolutePath());
                    return null;
                }
            }
            File tmp = File.createTempFile("img", "img");
            IOUtils.copy(imageFile.getInputstream(), new FileOutputStream(tmp));
            File thumbnailImage = new File(saveFld + File.separator + UUID.randomUUID().toString() + ".png");
            File fullResolution = new File(saveFld + File.separator + UUID.randomUUID().toString() + ".png");
            BufferedImage image = ImageIO.read(tmp);
            Image thumbnailIm = image.getScaledInstance(310, 210, Image.SCALE_SMOOTH);
            BufferedImage thumbnailBi = new BufferedImage(thumbnailIm.getWidth(null), thumbnailIm.getHeight(null), BufferedImage.TYPE_INT_RGB);
            Graphics bg = thumbnailBi.getGraphics();
            bg.drawImage(thumbnailIm, 0, 0, null);
            bg.dispose();
            ImageIO.write(thumbnailBi, "png", thumbnailImage);
            ImageIO.write(image, "png", fullResolution);
            if (!tmp.delete()) {
                logger.info("Unable to delete: " + tmp.getAbsolutePath());
            }
            String imageId = UUID.randomUUID().toString();
            imageBean.addImage(imageId, new ImageRecord(imageFile.getFileName(), fullResolution.getAbsolutePath(), thumbnailImage.getAbsolutePath(), userDisplay.getUser().getUsername()));
            return imageId;
        } catch (Throwable t) {
            logger.log(Level.SEVERE, "Unable to save the image.", t);
            return null;
        }
    }
```

## P090

**A**

```java
protected boolean update(String sql, int requiredRows, int maxRows) throws SQLException {
        if (LOG.isDebugEnabled()) {
            LOG.debug("executing " + sql + "...");
        }
        Connection connection = null;
        boolean oldAutoCommit = true;
        try {
            connection = dataSource.getConnection();
            connection.clearWarnings();
            oldAutoCommit = connection.getAutoCommit();
            connection.setAutoCommit(false);
            Statement statement = connection.createStatement();
            int rowsAffected = statement.executeUpdate(sql);
            if (requiredRows != -1 && rowsAffected < requiredRows) {
                LOG.warn("(" + rowsAffected + ") less than " + requiredRows + " rows affected, rolling back...");
                connection.rollback();
                return false;
            }
            if (maxRows != -1 && rowsAffected > maxRows) {
                LOG.warn("(" + rowsAffected + ") more than " + maxRows + " rows affected, rolling back...");
                connection.rollback();
                return false;
            }
            connection.commit();
            return true;
        } catch (SQLException e) {
            LOG.error("Unable to update database using: " + sql, e);
            throw e;
        } finally {
            try {
                if (connection != null) {
                    connection.setAutoCommit(oldAutoCommit);
                    connection.close();
                }
            } catch (SQLException e) {
                LOG.error("Unable to close connection: " + e, e);
            }
        }
    }
```

**B**

```java
public static int executeUpdate(EOAdaptorChannel channel, String sql, boolean autoCommit) throws SQLException {
        int rowsUpdated;
        boolean wasOpen = channel.isOpen();
        if (!wasOpen) {
            channel.openChannel();
        }
        Connection conn = ((JDBCContext) channel.adaptorContext()).connection();
        try {
            Statement stmt = conn.createStatement();
            try {
                rowsUpdated = stmt.executeUpdate(sql);
                if (autoCommit) {
                    conn.commit();
                }
            } catch (SQLException ex) {
                if (autoCommit) {
                    conn.rollback();
                }
                throw new RuntimeException("Failed to execute the statement '" + sql + "'.", ex);
            } finally {
                stmt.close();
            }
        } finally {
            if (!wasOpen) {
                channel.closeChannel();
            }
        }
        return rowsUpdated;
    }
```

## P091

**A**

```java
public ClientDTO changePassword(String pMail, String pMdp) {
        Client vClientBean = null;
        ClientDTO vClientDTO = null;
        vClientBean = mClientDao.getClient(pMail);
        if (vClientBean != null) {
            MessageDigest vMd5Instance;
            try {
                vMd5Instance = MessageDigest.getInstance("MD5");
                vMd5Instance.reset();
                vMd5Instance.update(pMdp.getBytes());
                byte[] vDigest = vMd5Instance.digest();
                BigInteger vBigInt = new BigInteger(1, vDigest);
                String vHashPassword = vBigInt.toString(16);
                vClientBean.setMdp(vHashPassword);
                vClientDTO = BeanToDTO.getInstance().createClientDTO(vClientBean);
            } catch (NoSuchAlgorithmException e) {
                e.printStackTrace();
            }
        }
        return vClientDTO;
    }
```

**B**

```java
private String[] verifyConnection(Socket clientConnection) throws Exception {
        List<String> requestLines = new ArrayList<String>();
        InputStream is = clientConnection.getInputStream();
        BufferedReader in = new BufferedReader(new InputStreamReader(is));
        StringTokenizer st = new StringTokenizer(in.readLine());
        if (!st.hasMoreTokens()) {
            throw new IllegalArgumentException("There's no method token in this connection");
        }
        String method = st.nextToken();
        if (!st.hasMoreTokens()) {
            throw new IllegalArgumentException("There's no URI token in this connection");
        }
        String uri = decodePercent(st.nextToken());
        if (!st.hasMoreTokens()) {
            throw new IllegalArgumentException("There's no version token in this connection");
        }
        String version = st.nextToken();
        Properties parms = new Properties();
        int qmi = uri.indexOf('?');
        if (qmi >= 0) {
            decodeParms(uri.substring(qmi + 1), parms);
            uri = decodePercent(uri.substring(0, qmi));
        }
        String params = "";
        if (parms.size() > 0) {
            params = "?";
            for (Object key : parms.keySet()) {
                params = params + key + "=" + parms.getProperty(((String) key)) + "&";
            }
            params = params.substring(0, params.length() - 1).replace(" ", "%20");
        }
        logger.debug("HTTP Request: " + method + " " + uri + params + " " + version);
        requestLines.add(method + " " + uri + params + " " + version);
        Properties headerVars = new Properties();
        String line;
        String currentBoundary = null;
        Stack<String> boundaryStack = new Stack<String>();
        boolean readingBoundary = false;
        String additionalData = "";
        while (in.ready() && (line = in.readLine()) != null) {
            if (line.equals("") && (headerVars.get("Content-Type") == null || headerVars.get("Content-Length") == null)) {
                break;
            }
            logger.debug("HTTP Request Header: " + line);
            if (line.contains(": ")) {
                String vals[] = line.split(": ");
                headerVars.put(vals[0].trim(), vals[1].trim());
            }
            if (!readingBoundary && line.contains(": ")) {
                if (line.contains("boundary=")) {
                    currentBoundary = line.split("boundary=")[1].trim();
                    boundaryStack.push("--" + currentBoundary);
                }
                continue;
            } else if (line.equals("") && boundaryStack.isEmpty()) {
                int val = Integer.parseInt((String) headerVars.get("Content-Length"));
                if (headerVars.getProperty("Content-Type").contains("x-www-form-urlencoded")) {
                    char buf[] = new char[val];
                    int read = in.read(buf);
                    line = String.valueOf(buf, 0, read);
                    additionalData = line;
                    logger.debug("HTTP Request Header Form Parameters: " + line);
                }
            } else if (line.equals(boundaryStack.peek()) && !readingBoundary) {
                readingBoundary = true;
            } else if (line.equals(boundaryStack.peek()) && readingBoundary) {
                readingBoundary = false;
            } else if (line.contains(": ") && readingBoundary) {
                if (method.equalsIgnoreCase("PUT")) {
                    if (line.contains("form-data; ")) {
                        String formValues = line.split("form-data; ")[1];
                        for (String varValue : formValues.replace("\"", "").split("; ")) {
                            String[] vV = varValue.split("=");
                            vV[0] = decodePercent(vV[0]);
                            vV[1] = decodePercent(vV[1]);
                            headerVars.put(vV[0], vV[1]);
                        }
                    }
                }
            } else if (line.contains("") && readingBoundary && !boundaryStack.isEmpty() && headerVars.get("filename") != null) {
                int length = Integer.parseInt(headerVars.getProperty("Content-Length"));
                if (headerVars.getProperty("Content-Transfer-Encoding").contains("binary")) {
                    File uploadFilePath = new File(VOctopusConfigurationManager.WebServerProperties.HTTPD_CONF.getPropertyValue("TempDirectory"));
                    if (!uploadFilePath.exists()) {
                        logger.error("Temporaty dir does not exist: " + uploadFilePath.getCanonicalPath());
                    }
                    if (!uploadFilePath.isDirectory()) {
                        logger.error("Temporary dir is not a directory: " + uploadFilePath.getCanonicalPath());
                    }
                    if (!uploadFilePath.canWrite()) {
                        logger.error("VOctopus Webserver doesn't have permissions to write on temporary dir: " + uploadFilePath.getCanonicalPath());
                    }
                    FileOutputStream out = null;
                    try {
                        String putUploadPath = uploadFilePath.getAbsolutePath() + "/" + headerVars.getProperty("filename");
                        out = new FileOutputStream(putUploadPath);
                        OutputStream outf = new BufferedOutputStream(out);
                        int c;
                        while (in.ready() && (c = in.read()) != -1 && length-- > 0) {
                            outf.write(c);
                        }
                    } finally {
                        if (out != null) {
                            out.close();
                        }
                    }
                    File copied = new File(VOctopusConfigurationManager.getInstance().getDocumentRootPath() + uri + headerVars.get("filename"));
                    File tempFile = new File(VOctopusConfigurationManager.WebServerProperties.HTTPD_CONF.getPropertyValue("TempDirectory") + "/" + headerVars.get("filename"));
                    FileChannel ic = new FileInputStream(tempFile.getAbsolutePath()).getChannel();
                    FileChannel oc = new FileOutputStream(copied.getAbsolutePath()).getChannel();
                    ic.transferTo(0, ic.size(), oc);
                    ic.close();
                    oc.close();
                }
            }
        }
        for (Object var : headerVars.keySet()) {
            requestLines.add(var + ": " + headerVars.get(var));
        }
        if (!additionalData.equals("")) {
            requestLines.add("ADDITIONAL" + additionalData);
        }
        return requestLines.toArray(new String[requestLines.size()]);
    }
```

## P092

**A**

```java
private void checkRoundtrip(byte[] content) throws Exception {
        InputStream in = new ByteArrayInputStream(content);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        CodecUtil.encodeQuotedPrintableBinary(in, out);
        in = new QuotedPrintableInputStream(new ByteArrayInputStream(out.toByteArray()));
        out = new ByteArrayOutputStream();
        IOUtils.copy(in, out);
        assertEquals(content, out.toByteArray());
    }
```

**B**

```java
@Test
    public void testStandardTee() throws Exception {
        final byte[] test = "test".getBytes();
        final InputStream source = new ByteArrayInputStream(test);
        final ByteArrayOutputStream destination1 = new ByteArrayOutputStream();
        final ByteArrayOutputStream destination2 = new ByteArrayOutputStream();
        final TeeOutputStream tee = new TeeOutputStream(destination1, destination2);
        org.apache.commons.io.IOUtils.copy(source, tee);
        tee.close();
        assertArrayEquals("the two arrays are equals", test, destination1.toByteArray());
        assertArrayEquals("the two arrays are equals", test, destination2.toByteArray());
        assertEquals("byte count", test.length, tee.getSize());
    }
```

## P093

**A**

```java
public Transaction() throws Exception {
        Connection Conn = null;
        Statement Stmt = null;
        try {
            Class.forName("org.gjt.mm.mysql.Driver").newInstance();
            Conn = DriverManager.getConnection(DBUrl);
            Conn.setAutoCommit(true);
            Stmt = Conn.createStatement();
            try {
                Stmt.executeUpdate("DROP TABLE trans_test");
            } catch (SQLException sqlEx) {
            }
            Stmt.executeUpdate("CREATE TABLE trans_test (id int not null primary key, decdata double) type=BDB");
            Conn.setAutoCommit(false);
            Stmt.executeUpdate("INSERT INTO trans_test (id, decdata) VALUES (1, 21.0)");
            Stmt.executeUpdate("INSERT INTO trans_test (id, decdata) VALUES (2, 23.485115)");
            Conn.rollback();
            System.out.println("Roll Ok");
            ResultSet RS = Stmt.executeQuery("SELECT * from trans_test");
            if (!RS.next()) {
                System.out.println("Ok");
            } else {
                System.out.println("Rollback failed");
            }
            Stmt.executeUpdate("INSERT INTO trans_test (id, decdata) VALUES (2, 23.485115)");
            Stmt.executeUpdate("INSERT INTO trans_test (id, decdata) VALUES (1, 21.485115)");
            Conn.commit();
            RS = Stmt.executeQuery("SELECT * from trans_test where id=2");
            if (RS.next()) {
                System.out.println(RS.getDouble(2));
                System.out.println("Ok");
            } else {
                System.out.println("Rollback failed");
            }
        } catch (Exception ex) {
            throw ex;
        } finally {
            if (Stmt != null) {
                try {
                    Stmt.close();
                } catch (SQLException SQLEx) {
                }
            }
            if (Conn != null) {
                try {
                    Conn.close();
                } catch (SQLException SQLEx) {
                }
            }
        }
    }
```

**B**

```java
private void insert() throws SQLException, NamingException {
        Logger logger = getLogger();
        if (logger.isDebugEnabled()) {
            logger.debug("enter - " + getClass().getName() + ".insert()");
        }
        try {
            if (logger.isInfoEnabled()) {
                logger.info("insert(): Create new sequencer record for " + getName());
            }
            Connection conn = null;
            PreparedStatement stmt = null;
            ResultSet rs = null;
            try {
                InitialContext ctx = new InitialContext();
                DataSource ds = (DataSource) ctx.lookup(dataSourceName);
                conn = ds.getConnection();
                conn.setReadOnly(false);
                stmt = conn.prepareStatement(INSERT_SEQ);
                stmt.setString(INS_NAME, getName());
                stmt.setLong(INS_NEXT_KEY, defaultInterval * 2);
                stmt.setLong(INS_INTERVAL, defaultInterval);
                stmt.setLong(INS_UPDATE, System.currentTimeMillis());
                try {
                    if (stmt.executeUpdate() != 1) {
                        nextId = -1L;
                        logger.warn("insert(): Failed to create sequencer entry for " + getName() + " (no error message)");
                    } else if (logger.isInfoEnabled()) {
                        nextId = defaultInterval;
                        nextSeed = defaultInterval * 2;
                        interval = defaultInterval;
                        logger.info("insert(): First ID will be " + nextId);
                    }
                } catch (SQLException e) {
                    logger.warn("insert(): Error inserting row into database, possible concurrency issue: " + e.getMessage());
                    if (logger.isDebugEnabled()) {
                        e.printStackTrace();
                    }
                    nextId = -1L;
                }
                if (!conn.getAutoCommit()) {
                    conn.commit();
                }
            } finally {
                if (rs != null) {
                    try {
                        rs.close();
                    } catch (SQLException ignore) {
                    }
                }
                if (stmt != null) {
                    try {
                        stmt.close();
                    } catch (SQLException ignore) {
                    }
                }
                if (conn != null) {
                    if (!conn.getAutoCommit()) {
                        try {
                            conn.rollback();
                        } catch (SQLException ignore) {
                        }
                    }
                    try {
                        conn.close();
                    } catch (SQLException ignore) {
                    }
                }
            }
        } finally {
            if (logger.isDebugEnabled()) {
                logger.debug("exit - " + getClass().getName() + ".insert()");
            }
        }
    }
```

## P094

**A**

```java
private static void doCopyFile(File srcFile, File destFile, boolean preserveFileDate) throws IOException {
        if (destFile.exists() && destFile.isDirectory()) {
            throw new IOException("Destination '" + destFile + "' exists but is a directory");
        }
        FileInputStream input = new FileInputStream(srcFile);
        try {
            FileOutputStream output = new FileOutputStream(destFile);
            try {
                IOUtils.copy(input, output);
            } finally {
                IOUtils.close(output);
            }
        } finally {
            IOUtils.close(input);
        }
        if (srcFile.length() != destFile.length()) {
            throw new IOException("Failed to copy full contents from '" + srcFile + "' to '" + destFile + "'");
        }
        if (preserveFileDate) {
            destFile.setLastModified(srcFile.lastModified());
        }
    }
```

**B**

```java
private static void doCopyFile(File srcFile, File destFile, boolean preserveFileDate) throws IOException {
        if (destFile.exists() && destFile.isDirectory()) {
            throw new IOException("Destination '" + destFile + "' exists but is a directory");
        }
        FileInputStream input = new FileInputStream(srcFile);
        try {
            FileOutputStream output = new FileOutputStream(destFile);
            try {
                IOUtils.copy(input, output);
            } finally {
                IOUtils.closeQuietly(output);
            }
        } finally {
            IOUtils.closeQuietly(input);
        }
        if (srcFile.length() != destFile.length()) {
            throw new IOException("Failed to copy full contents from '" + srcFile + "' to '" + destFile + "'");
        }
        if (preserveFileDate) {
            destFile.setLastModified(srcFile.lastModified());
        }
    }
```

## P095

**A**

```java
private void doTask() {
        try {
            log("\n\n\n\n\n\n\n\n\n");
            log(" =================================================");
            log(" = Starting PSCafePOS                            =");
            log(" =================================================");
            log(" =   An open source point of sale system         =");
            log(" =   for educational organizations.              =");
            log(" =================================================");
            log(" = General Information                           =");
            log(" =   http://pscafe.sourceforge.net               =");
            log(" = Free Product Support                          =");
            log(" =   http://www.sourceforge.net/projects/pscafe  =");
            log(" =================================================");
            log(" = License Overview                              =");
            log(" =================================================");
            log(" = PSCafePOS is a POS System for Schools         =");
            log(" = Copyright (C) 2007 Charles Syperski           =");
            log(" =                                               =");
            log(" = This program is free software; you can        =");
            log(" = redistribute it and/or modify it under the    =");
            log(" = terms of the GNU General Public License as    =");
            log(" = published by the Free Software Foundation;    =");
            log(" = either version 2 of the License, or any later =");
            log(" = version.                                      =");
            log(" =                                               =");
            log(" = This program is distributed in the hope that  =");
            log(" = it will be useful, but WITHOUT ANY WARRANTY;  =");
            log(" = without even the implied warranty of          =");
            log(" = MERCHANTABILITY or FITNESS FOR A PARTICULAR   =");
            log(" = PURPOSE.                                      =");
            log(" =                                               =");
            log(" = See the GNU General Public License for more   =");
            log(" = details.                                      =");
            log(" =                                               =");
            log(" = You should have received a copy of the GNU    =");
            log(" = General Public License along with this        =");
            log(" = program; if not, write to the                 =");
            log(" =                                               =");
            log(" =      Free Software Foundation, Inc.           =");
            log(" =      59 Temple Place, Suite 330               =");
            log(" =      Boston, MA 02111-1307 USA                =");
            log(" =================================================");
            log(" = If you have any questions of comments please  =");
            log(" = let us know at http://pscafe.sourceforge.net  =");
            log(" =================================================");
            pause();
            File settings;
            if (blAltSettings) {
                System.out.println("\n  + Alternative path specified at run time:");
                System.out.println("    Path: " + strAltPath);
                settings = new File(strAltPath);
            } else {
                settings = new File("etc" + File.separatorChar + "settings.dbp");
            }
            System.out.print("\n  + Checking for existance of settings...");
            boolean blGo = false;
            if (settings.exists() && settings.canRead()) {
                log("[OK]");
                blGo = true;
                if (forceConfig) {
                    System.out.print("\n  + Running Config Wizard (at user request)...");
                    Process pp = Runtime.getRuntime().exec("java -cp . PSSettingWizard etc" + File.separatorChar + "settings.dbp");
                    InputStream stderr = pp.getErrorStream();
                    InputStream stdin = pp.getInputStream();
                    InputStreamReader isr = new InputStreamReader(stdin);
                    BufferedReader br = new BufferedReader(isr);
                    String ln = null;
                    while ((ln = br.readLine()) != null) System.out.println("  " + ln);
                    pp.waitFor();
                }
            } else {
                log("[FAILED]");
                settings = new File("etc" + File.separatorChar + "settings.dbp.firstrun");
                System.out.print("\n  + Checking if this is the first run... ");
                if (settings.exists() && settings.canRead()) {
                    log("[FOUND]");
                    File toFile = new File("etc" + File.separatorChar + "settings.dbp");
                    FileInputStream from = null;
                    FileOutputStream to = null;
                    try {
                        from = new FileInputStream(settings);
                        to = new FileOutputStream(toFile);
                        byte[] buffer = new byte[4096];
                        int bytes_read;
                        while ((bytes_read = from.read(buffer)) != -1) {
                            to.write(buffer, 0, bytes_read);
                        }
                        if (toFile.exists() && toFile.canRead()) {
                            settings = null;
                            settings = new File("etc" + File.separatorChar + "settings.dbp");
                        }
                        System.out.print("\n  + Running Settings Wizard... ");
                        try {
                            Process p = Runtime.getRuntime().exec("java PSSettingWizard etc" + File.separatorChar + "settings.dbp");
                            InputStream stderr = p.getErrorStream();
                            InputStream stdin = p.getInputStream();
                            InputStreamReader isr = new InputStreamReader(stdin);
                            BufferedReader br = new BufferedReader(isr);
                            String ln = null;
                            while ((ln = br.readLine()) != null) System.out.println("  " + ln);
                            p.waitFor();
                            log("[OK]");
                            if (p.exitValue() == 0) blGo = true;
                        } catch (InterruptedException i) {
                            System.err.println(i.getMessage());
                        }
                    } catch (Exception ex) {
                        System.err.println(ex.getMessage());
                    } finally {
                        if (from != null) try {
                            from.close();
                        } catch (IOException e) {
                            ;
                        }
                        if (to != null) try {
                            to.close();
                        } catch (IOException e) {
                            ;
                        }
                    }
                } else {
                    settings = null;
                    settings = new File("etc" + File.separatorChar + "settings.dbp");
                    DBSettingsWriter writ = new DBSettingsWriter();
                    writ.writeFile(new DBSettings(), settings);
                    blGo = true;
                }
            }
            if (blGo) {
                String cp = ".";
                try {
                    File classpath = new File("lib");
                    File[] subFiles = classpath.listFiles();
                    for (int i = 0; i < subFiles.length; i++) {
                        if (subFiles[i].isFile()) {
                            cp += File.pathSeparatorChar + "lib" + File.separatorChar + subFiles[i].getName() + "";
                        }
                    }
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
                try {
                    boolean blExecutePOS = false;
                    System.out.print("\n  + Checking runtime settings...         ");
                    DBSettings info = null;
                    if (settings == null) settings = new File("etc" + File.separatorChar + "settings.dbp");
                    if (settings.exists() && settings.canRead()) {
                        DBSettingsWriter writ = new DBSettingsWriter();
                        info = (DBSettings) writ.loadSettingsDB(settings);
                        if (info != null) {
                            blExecutePOS = true;
                        }
                    }
                    if (blExecutePOS) {
                        log("[OK]");
                        String strSSL = "";
                        String strSSLDebug = "";
                        if (info != null) {
                            debug = info.get(DBSettings.MAIN_DEBUG).compareTo("1") == 0;
                            if (debug) log("       * Debug Mode is ON"); else log("       * Debug Mode is OFF");
                            if (info.get(DBSettings.POS_SSLENABLED).compareTo("1") == 0) {
                                strSSL = "-Djavax.net.ssl.keyStore=" + info.get(DBSettings.POS_SSLKEYSTORE) + " -Djavax.net.ssl.keyStorePassword=pscafe -Djavax.net.ssl.trustStore=" + info.get(DBSettings.POS_SSLTRUSTSTORE) + " -Djavax.net.ssl.trustStorePassword=pscafe";
                                log("       * Using SSL");
                                debug("            " + strSSL);
                                if (info.get(DBSettings.POS_SSLDEBUG).compareTo("1") == 0) {
                                    strSSLDebug = "-Djavax.net.debug=all";
                                    log("       * SSL Debugging enabled");
                                    debug("            " + strSSLDebug);
                                }
                            }
                        }
                        String strPOSRun = "java  -cp " + cp + " " + strSSL + " " + strSSLDebug + " POSDriver " + settings.getPath();
                        debug(strPOSRun);
                        System.out.print("\n  + Running PSCafePOS...                 ");
                        Process pr = Runtime.getRuntime().exec(strPOSRun);
                        System.out.print("[OK]\n\n");
                        InputStream stderr = pr.getErrorStream();
                        InputStream stdin = pr.getInputStream();
                        InputStreamReader isr = new InputStreamReader(stdin);
                        InputStreamReader isre = new InputStreamReader(stderr);
                        BufferedReader br = new BufferedReader(isr);
                        BufferedReader bre = new BufferedReader(isre);
                        String line = null;
                        String lineError = null;
                        log(" =================================================");
                        log(" =        Output from PSCafePOS System           =");
                        log(" =================================================");
                        while ((line = br.readLine()) != null || (lineError = bre.readLine()) != null) {
                            if (line != null) System.out.println(" [PSCafePOS]" + line);
                            if (lineError != null) System.out.println(" [ERR]" + lineError);
                        }
                        pr.waitFor();
                        log(" =================================================");
                        log(" =       End output from PSCafePOS System        =");
                        log(" =              PSCafePOS has exited             =");
                        log(" =================================================");
                    } else {
                        log("[Failed]");
                    }
                } catch (Exception i) {
                    log(i.getMessage());
                    i.printStackTrace();
                }
            }
        } catch (Exception e) {
            log(e.getMessage());
        }
    }
```

**B**

```java
public static void testAutoIncrement() {
        final int count = 3;
        final Object lock = new Object();
        for (int i = 0; i < count; i++) {
            new Thread(new Runnable() {

                @Override
                public void run() {
                    while (true) {
                        StringBuilder buffer = new StringBuilder(128);
                        buffer.append("insert into DOMAIN (                         ").append(LS);
                        buffer.append("    DOMAIN_ID, TOP_DOMAIN_ID, DOMAIN_HREF,   ").append(LS);
                        buffer.append("    DOMAIN_RANK, DOMAIN_TYPE, DOMAIN_STATUS, ").append(LS);
                        buffer.append("    DOMAIN_ICO_CREATED, DOMAIN_CDATE         ").append(LS);
                        buffer.append(") values (                   ").append(LS);
                        buffer.append("    null ,null, ?,").append(LS);
                        buffer.append("    1, 2, 1,                 ").append(LS);
                        buffer.append("    0, now()                 ").append(LS);
                        buffer.append(")                            ").append(LS);
                        String sqlInsert = buffer.toString();
                        boolean isAutoCommit = false;
                        int i = 0;
                        Connection conn = null;
                        PreparedStatement pstmt = null;
                        ResultSet rs = null;
                        try {
                            conn = ConnHelper.getConnection();
                            conn.setAutoCommit(isAutoCommit);
                            pstmt = conn.prepareStatement(sqlInsert);
                            for (i = 0; i < 10; i++) {
                                String lock = "" + ((int) (Math.random() * 100000000)) % 100;
                                pstmt.setString(1, lock);
                                pstmt.executeUpdate();
                            }
                            if (!isAutoCommit) conn.commit();
                            rs = pstmt.executeQuery("select max(DOMAIN_ID) from DOMAIN");
                            if (rs.next()) {
                                String str = System.currentTimeMillis() + " " + rs.getLong(1);
                            }
                        } catch (Exception e) {
                            try {
                                if (!isAutoCommit) conn.rollback();
                            } catch (SQLException ex) {
                                ex.printStackTrace(System.out);
                            }
                            String msg = System.currentTimeMillis() + " " + Thread.currentThread().getName() + " - " + i + " " + e.getMessage() + LS;
                            FileIO.writeToFile("D:/DEAD_LOCK.txt", msg, true, "GBK");
                        } finally {
                            ConnHelper.close(conn, pstmt, rs);
                        }
                    }
                }
            }).start();
        }
    }
```

## P096

**A**

```java
PackageFileImpl(PackageDirectoryImpl dir, String name, InputStream data) throws IOException {
        this.dir = dir;
        this.name = name;
        this.updates = dir.getUpdates();
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        IOUtils.copy(data, stream);
        updates.setNewData(getFullName(), stream.toByteArray());
        stream.close();
    }
```

**B**

```java
private static void processFile(String file) throws IOException {
        FileInputStream in = new FileInputStream(file);
        int read = 0;
        byte[] buf = new byte[2048];
        ByteArrayOutputStream bout = new ByteArrayOutputStream();
        while ((read = in.read(buf)) > 0) bout.write(buf, 0, read);
        in.close();
        String converted = bout.toString().replaceAll("@project.name@", projectNameS).replaceAll("@base.package@", basePackageS).replaceAll("@base.dir@", baseDir).replaceAll("@webapp.dir@", webAppDir).replaceAll("path=\"target/classes\"", "path=\"src/main/webapp/WEB-INF/classes\"");
        FileOutputStream out = new FileOutputStream(file);
        out.write(converted.getBytes());
        out.close();
    }
```

## P097

**A**

```java
private void exportJar(File root, List<File> list, Manifest manifest) throws Exception {
        JarOutputStream jarOut = null;
        FileInputStream fin = null;
        try {
            jarOut = new JarOutputStream(new FileOutputStream(jarFile), manifest);
            for (int i = 0; i < list.size(); i++) {
                String filename = list.get(i).getAbsolutePath();
                filename = filename.substring(root.getAbsolutePath().length() + 1);
                fin = new FileInputStream(list.get(i));
                JarEntry entry = new JarEntry(filename.replace('\\', '/'));
                jarOut.putNextEntry(entry);
                byte[] buf = new byte[4096];
                int read;
                while ((read = fin.read(buf)) != -1) {
                    jarOut.write(buf, 0, read);
                }
                jarOut.closeEntry();
                jarOut.flush();
            }
        } finally {
            if (fin != null) {
                try {
                    fin.close();
                } catch (Exception e) {
                    ExceptionOperation.operate(e);
                }
            }
            if (jarOut != null) {
                try {
                    jarOut.close();
                } catch (Exception e) {
                }
            }
        }
    }
```

**B**

```java
public static Class[] findSubClasses(Class baseClass) {
        String packagePath = "/" + baseClass.getPackage().getName().replace('.', '/');
        URL url = baseClass.getResource(packagePath);
        if (url == null) {
            return new Class[0];
        }
        List<Class> derivedClasses = new ArrayList<Class>();
        try {
            URLConnection connection = url.openConnection();
            if (connection instanceof JarURLConnection) {
                JarFile jarFile = ((JarURLConnection) connection).getJarFile();
                Enumeration e = jarFile.entries();
                while (e.hasMoreElements()) {
                    ZipEntry entry = (ZipEntry) e.nextElement();
                    String entryName = entry.getName();
                    if (entryName.endsWith(".class")) {
                        String clazzName = entryName.substring(0, entryName.length() - 6);
                        clazzName = clazzName.replace('/', '.');
                        try {
                            Class clazz = Class.forName(clazzName);
                            if (isConcreteSubclass(baseClass, clazz)) {
                                derivedClasses.add(clazz);
                            }
                        } catch (Throwable ignoreIt) {
                        }
                    }
                }
            } else if (connection instanceof FileURLConnection) {
                File file = new File(url.getFile());
                File[] files = file.listFiles();
                for (int i = 0; i < files.length; i++) {
                    String filename = files[i].getName();
                    if (filename.endsWith(".class")) {
                        filename = filename.substring(0, filename.length() - 6);
                        String clazzname = baseClass.getPackage().getName() + "." + filename;
                        try {
                            Class clazz = Class.forName(clazzname);
                            if (isConcreteSubclass(baseClass, clazz)) {
                                derivedClasses.add(clazz);
                            }
                        } catch (Throwable ignoreIt) {
                        }
                    }
                }
            }
        } catch (IOException ignoreIt) {
        }
        return derivedClasses.toArray(new Class[derivedClasses.size()]);
    }
```

## P098

**A**

```java
public static File copyFile(String path) {
        File src = new File(path);
        File dest = new File(src.getName());
        try {
            if (!dest.exists()) {
                dest.createNewFile();
            }
            FileChannel source = new FileInputStream(src).getChannel();
            FileChannel destination = new FileOutputStream(dest).getChannel();
            destination.transferFrom(source, 0, source.size());
            source.close();
            destination.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return dest;
    }
```

**B**

```java
public void copyFile(File sourceFile, File destFile) throws IOException {
        if (!destFile.exists()) {
            destFile.createNewFile();
        }
        FileChannel source = null;
        FileChannel destination = null;
        Closer c = new Closer();
        try {
            source = c.register(new FileInputStream(sourceFile).getChannel());
            destination = c.register(new FileOutputStream(destFile).getChannel());
            destination.transferFrom(source, 0, source.size());
        } catch (IOException e) {
            c.doNotThrow();
            throw e;
        } finally {
            c.closeAll();
        }
    }
```

## P099

**A**

```java
public void run() {
            try {
                File outDir = new File(outDirTextField.getText());
                if (!outDir.exists()) {
                    SwingUtilities.invokeLater(new Runnable() {

                        public void run() {
                            JOptionPane.showMessageDialog(UnpackWizard.this, "The chosen directory does not exist!", "Directory Not Found Error", JOptionPane.ERROR_MESSAGE);
                        }
                    });
                    return;
                }
                if (!outDir.isDirectory()) {
                    SwingUtilities.invokeLater(new Runnable() {

                        public void run() {
                            JOptionPane.showMessageDialog(UnpackWizard.this, "The chosen file is not a directory!", "Not a Directory Error", JOptionPane.ERROR_MESSAGE);
                        }
                    });
                    return;
                }
                if (!outDir.canWrite()) {
                    SwingUtilities.invokeLater(new Runnable() {

                        public void run() {
                            JOptionPane.showMessageDialog(UnpackWizard.this, "Cannot write to the chosen directory!", "Directory Not Writeable Error", JOptionPane.ERROR_MESSAGE);
                        }
                    });
                    return;
                }
                File archiveDir = new File("foo.bar").getAbsoluteFile().getParentFile();
                URL baseUrl = UnpackWizard.class.getClassLoader().getResource(UnpackWizard.class.getName().replaceAll("\\.", "/") + ".class");
                if (baseUrl.getProtocol().equals("jar")) {
                    String jarPath = baseUrl.getPath();
                    jarPath = jarPath.substring(0, jarPath.indexOf('!'));
                    if (jarPath.startsWith("file:")) {
                        try {
                            archiveDir = new File(new URI(jarPath)).getAbsoluteFile().getParentFile();
                        } catch (URISyntaxException e1) {
                            e1.printStackTrace(System.err);
                        }
                    }
                }
                SortedMap<Integer, String> inputFileNames = new TreeMap<Integer, String>();
                for (Entry<Object, Object> anEntry : indexProperties.entrySet()) {
                    String key = anEntry.getKey().toString();
                    if (key.startsWith("archive file ")) {
                        inputFileNames.put(Integer.parseInt(key.substring("archive file ".length())), anEntry.getValue().toString());
                    }
                }
                byte[] buff = new byte[64 * 1024];
                try {
                    long bytesToWrite = 0;
                    long bytesReported = 0;
                    long bytesWritten = 0;
                    for (String aFileName : inputFileNames.values()) {
                        File aFile = new File(archiveDir, aFileName);
                        if (aFile.exists()) {
                            if (aFile.isFile()) {
                                bytesToWrite += aFile.length();
                            } else {
                                final File wrongFile = aFile;
                                SwingUtilities.invokeLater(new Runnable() {

                                    public void run() {
                                        JOptionPane.showMessageDialog(UnpackWizard.this, "File \"" + wrongFile.getAbsolutePath() + "\" is not a standard file!", "Non Standard File Error", JOptionPane.ERROR_MESSAGE);
                                    }
                                });
                                return;
                            }
                        } else {
                            final File wrongFile = aFile;
                            SwingUtilities.invokeLater(new Runnable() {

                                public void run() {
                                    JOptionPane.showMessageDialog(UnpackWizard.this, "File \"" + wrongFile.getAbsolutePath() + "\" does not exist!", "File Not Found Error", JOptionPane.ERROR_MESSAGE);
                                }
                            });
                            return;
                        }
                    }
                    MultiFileInputStream mfis = new MultiFileInputStream(archiveDir, inputFileNames.values().toArray(new String[inputFileNames.size()]));
                    TarArchiveInputStream tis = new TarArchiveInputStream(new BufferedInputStream(mfis));
                    TarArchiveEntry tarEntry = tis.getNextTarEntry();
                    while (tarEntry != null) {
                        File outFile = new File(outDir.getAbsolutePath() + "/" + tarEntry.getName());
                        if (outFile.exists()) {
                            final File wrongFile = outFile;
                            SwingUtilities.invokeLater(new Runnable() {

                                public void run() {
                                    JOptionPane.showMessageDialog(UnpackWizard.this, "Was about to write out file \"" + wrongFile.getAbsolutePath() + "\" but it already " + "exists.\nPlease [re]move existing files out of the way " + "and try again.", "File Not Found Error", JOptionPane.ERROR_MESSAGE);
                                }
                            });
                            return;
                        }
                        if (tarEntry.isDirectory()) {
                            outFile.getAbsoluteFile().mkdirs();
                        } else {
                            outFile.getAbsoluteFile().getParentFile().mkdirs();
                            OutputStream os = new BufferedOutputStream(new FileOutputStream(outFile));
                            int len = tis.read(buff, 0, buff.length);
                            while (len != -1) {
                                os.write(buff, 0, len);
                                bytesWritten += len;
                                if (bytesWritten - bytesReported > (10 * 1024 * 1024)) {
                                    bytesReported = bytesWritten;
                                    final int progress = (int) (bytesReported * 100 / bytesToWrite);
                                    SwingUtilities.invokeLater(new Runnable() {

                                        @Override
                                        public void run() {
                                            progressBar.setValue(progress);
                                        }
                                    });
                                }
                                len = tis.read(buff, 0, buff.length);
                            }
                            os.close();
                        }
                        tarEntry = tis.getNextTarEntry();
                    }
                    long expectedCrc = 0;
                    try {
                        expectedCrc = Long.parseLong(indexProperties.getProperty("CRC32", "0"));
                    } catch (NumberFormatException e) {
                        System.err.println("Error while obtaining the expected CRC");
                        e.printStackTrace(System.err);
                    }
                    if (mfis.getCRC() == expectedCrc) {
                        SwingUtilities.invokeLater(new Runnable() {

                            @Override
                            public void run() {
                                progressBar.setValue(0);
                                JOptionPane.showMessageDialog(UnpackWizard.this, "Extraction completed successfully!", "Done!", JOptionPane.INFORMATION_MESSAGE);
                            }
                        });
                        return;
                    } else {
                        System.err.println("CRC Error: was expecting " + expectedCrc + " but got " + mfis.getCRC());
                        SwingUtilities.invokeLater(new Runnable() {

                            public void run() {
                                progressBar.setValue(0);
                                JOptionPane.showMessageDialog(UnpackWizard.this, "CRC Error: the data extracted does not have the expected CRC!\n" + "You should probably delete the extracted files, as they are " + "likely to be invalid.", "CRC Error", JOptionPane.ERROR_MESSAGE);
                            }
                        });
                        return;
                    }
                } catch (final IOException e) {
                    e.printStackTrace(System.err);
                    SwingUtilities.invokeLater(new Runnable() {

                        public void run() {
                            progressBar.setValue(0);
                            JOptionPane.showMessageDialog(UnpackWizard.this, "Input/Output Error: " + e.getLocalizedMessage(), "Input/Output Error", JOptionPane.ERROR_MESSAGE);
                        }
                    });
                    return;
                }
            } finally {
                SwingUtilities.invokeLater(new Runnable() {

                    public void run() {
                        progressBar.setValue(0);
                        setEnabled(true);
                    }
                });
            }
        }
```

**B**

```java
private void loadMap() {
        final String wordList = "vietwordlist.txt";
        try {
            File dataFile = new File(supportDir, wordList);
            if (!dataFile.exists()) {
                final ReadableByteChannel input = Channels.newChannel(ClassLoader.getSystemResourceAsStream("dict/" + dataFile.getName()));
                final FileChannel output = new FileOutputStream(dataFile).getChannel();
                output.transferFrom(input, 0, 1000000L);
                input.close();
                output.close();
            }
            long fileLastModified = dataFile.lastModified();
            if (map == null) {
                map = new HashMap<String, String>();
            } else {
                if (fileLastModified <= mapLastModified) {
                    return;
                }
                map.clear();
            }
            mapLastModified = fileLastModified;
            BufferedReader bs = new BufferedReader(new InputStreamReader(new FileInputStream(dataFile), "UTF-8"));
            String accented;
            while ((accented = bs.readLine()) != null) {
                String plain = VietUtilities.stripDiacritics(accented);
                map.put(plain.toLowerCase(), accented);
            }
            bs.close();
        } catch (IOException e) {
            map = null;
            e.printStackTrace();
            JOptionPane.showMessageDialog(this, myResources.getString("Cannot_find_\"") + wordList + myResources.getString("\"_in\n") + supportDir.toString(), VietPad.APP_NAME, JOptionPane.ERROR_MESSAGE);
        }
    }
```

## P100

**A**

```java
private String getStoreName() {
        try {
            final MessageDigest digest = MessageDigest.getInstance("MD5");
            digest.update(protectionDomain.getBytes());
            final byte[] bs = digest.digest();
            final StringBuffer sb = new StringBuffer(bs.length * 2);
            for (int i = 0; i < bs.length; i++) {
                final String s = Integer.toHexString(bs[i] & 0xff);
                if (s.length() < 2) sb.append('0');
                sb.append(s);
            }
            return sb.toString();
        } catch (final NoSuchAlgorithmException e) {
            throw new RuntimeException("Can't save credentials: digest method MD5 unavailable.");
        }
    }
```

**B**

```java
private String sha1(String s) {
        String encrypt = s;
        try {
            MessageDigest sha = MessageDigest.getInstance("SHA-1");
            sha.update(s.getBytes());
            byte[] digest = sha.digest();
            final StringBuffer buffer = new StringBuffer();
            for (int i = 0; i < digest.length; ++i) {
                final byte b = digest[i];
                final int value = (b & 0x7F) + (b < 0 ? 128 : 0);
                buffer.append(value < 16 ? "0" : "");
                buffer.append(Integer.toHexString(value));
            }
            encrypt = buffer.toString();
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        }
        return encrypt;
    }
```
